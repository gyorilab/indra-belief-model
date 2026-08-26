"""Tests for the processed-shard vLLM runner."""
from __future__ import annotations

import gzip
import importlib.util
import json
import re
import sys
import threading
import time
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
    """An OBSERVED gemma reply: disconfirm fields, one per line, no punctuation.

    The fixture is a reply from the vLLM-served model this runner drives, which
    makes it worth keeping. `indra_belief.verdict` is the SINGLE parser for the
    live path and batch replay, including the two patterns that make this reply
    readable.
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
    identically on the live scorer and the batch replay, because all three read
    through `indra_belief.verdict`.
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
    """Prompt bytes must match the live scorer for every variant.

    A calibration profile is keyed on (model, prompt sha), so one drifted
    character scores against a profile fitted for a prompt that was never sent.

    Pinning one constant establishes parity only for that variant and freezes
    the CHOICE of variant. PARITY is the property; choice is a flag.

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
    """The corpus-scale default explicitly disables chain of thought.

    At 60M evidences an unasked-for CoT is the entire bill, so the request must
    carry explicit reasoning control rather than accept the chat template's
    default.
    """
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

    The calibration banner reads `DISCONFIRM_SYSTEM_PROMPT`, imported INSIDE
    `MonolithicPrompt.__init__` to keep this file importable without the scorer's
    dependency graph. A module-scope reference there dies with NameError before
    touching a shard, while the rest of the suite exercises only helpers.

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
    """An unfitted (model, prompt) must refuse rather than quietly publish
    hard-gate beliefs at corpus scale.

    The unfitted pair is chosen EXPLICITLY and its premise is ASSERTED, so
    fitting it makes this test fail as stale instead of quietly going hollow.

    `disconfirm_relnature_rf` is the deliberated contract. It is fitted on
    Bedrock and on MLX but not here, and not by oversight: its verdict lands
    ~56 tokens deep, so it cannot emit the in-call margin the corpus path reads.
    """
    import hashlib
    import sys

    import pytest as _pytest

    from indra_belief.calibration_constants import calibration_for
    from indra_belief.scorers.monolithic import scorer as mono

    unfitted_variant = "disconfirm_relnature_rf"
    sha = hashlib.sha256(
        mono.VARIANTS[unfitted_variant].system_prompt.encode("utf-8")
    ).hexdigest()
    assert calibration_for("vllm-gemma-4-26b", prompt_sha256=sha) is None, (
        f"premise is stale: {unfitted_variant} is now fitted for "
        "vllm-gemma-4-26b, so this test no longer exercises the refusal. "
        "Pick a pair that is genuinely unfitted, or delete the test knowingly."
    )

    empty = tmp_path / "shards"
    empty.mkdir()
    monkeypatch.setattr(sys, "argv", [
        "run_vllm_processed_shards.py",
        "--input-dir", str(empty),
        "--output-dir", str(tmp_path / "out"),
        "--model", "vllm-local",
        "--variant", unfitted_variant,
        "--require-calibrated",
    ])
    with _pytest.raises(SystemExit) as excinfo:
        runner.main()
    assert "no ship-approved profile" in str(excinfo.value)


def test_the_shipped_default_variant_is_fitted_for_the_corpus_stack(tmp_path, monkeypatch):
    """The other half: the pair the 60M run actually uses must PASS the gate.

    Without this, the refusal test above could be satisfied by a runner that
    refuses everything.
    """
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
    assert "no shards found" in str(excinfo.value), (
        "the corpus stack's own (model, prompt) pair no longer resolves a "
        f"ship-approved profile; died with: {excinfo.value}"
    )


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


# ── the offline dispatcher ────────────────────────────────────────────────────
#
# Everything below stubs `vllm` through sys.modules, which makes the batching
# thread exercisable without a GPU.
#
# WHAT THAT DOES NOT BUY. The stub is written here, so these tests pin OUR
# dispatcher against shapes THIS FILE defines -- they cannot tell us that real
# vLLM returns those shapes. No machine in this repo's reach can load a 26B
# model, so the offline backend remains unexercised against the engine itself.
# The corpus path runs --backend server.


class _StubSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _StubLogprob:
    def __init__(self, token, value):
        self.decoded_token = token
        self.logprob = value


class _StubCompletion:
    text = "correct\nhigh"
    token_ids = [1]
    logprobs = [{1: _StubLogprob("correct", -0.1), 2: _StubLogprob("incorrect", -2.1)}]
    finish_reason = "stop"


class _StubOutput:
    outputs = [_StubCompletion()]
    prompt_token_ids = [10, 11]


class _PreemptedOutput:
    """What an aborted or preempted request looks like: no completion at all."""

    outputs: list = []
    prompt_token_ids = [10, 11]


class _Caller(threading.Thread):
    """One `post`, with its outcome captured instead of raised in the worker."""

    def __init__(self, client, payload):
        super().__init__(daemon=True)
        self.client = client
        self.payload = payload
        self.response = None
        self.error = None

    def run(self):
        try:
            self.response = self.client.post("", json=self.payload)
        except BaseException as exc:  # noqa: BLE001 - the outcome under test
            self.error = exc


def _install_vllm(monkeypatch, chat, *, structured=None, calls=None,
                  instances=None, gate=None):
    """Stub `vllm` with an LLM whose chat() is `chat(conversations)`."""

    class LLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            if instances is not None:
                instances.append(self)

        def chat(self, conversations, sampling_params, use_tqdm, **kwargs):
            if calls is not None:
                calls.append({"conversations": conversations,
                              "sampling_params": sampling_params,
                              "kwargs": kwargs})
            if gate is not None:
                # Holds the FIRST call, so the requests posted while it is held
                # are guaranteed to reach the dispatcher as ONE batch.
                gate.wait(5)
            return chat(conversations)

    module = types.SimpleNamespace(LLM=LLM, SamplingParams=_StubSamplingParams)
    monkeypatch.setitem(sys.modules, "vllm", module)
    if structured is not None:
        monkeypatch.setitem(sys.modules, "vllm.sampling_params", structured)
        module.sampling_params = structured
    else:
        monkeypatch.setitem(sys.modules, "vllm.sampling_params",
                            types.SimpleNamespace())
    return module


def _batched(monkeypatch, chat, payload_a, payload_b, **kwargs):
    """Send two payloads through ONE batch and return their callers.

    The 2 ms collection window is a race the test must not depend on, so the
    first chat() call is held open while both payloads are queued behind it.
    """
    gate = threading.Event()
    calls: list = []
    _install_vllm(monkeypatch, chat, calls=calls, gate=gate, **kwargs)
    client = runner.OfflineVllmClient(
        "model/path", batch_size=8, gpu_memory_utilization=0.8, timeout=5
    )
    warm = _Caller(client, {"messages": [{"role": "user", "content": "warm"}],
                            "max_tokens": 4, "temperature": 0})
    warm.start()
    time.sleep(0.05)
    first = _Caller(client, payload_a)
    first.start()
    time.sleep(0.05)
    second = _Caller(client, payload_b)
    second.start()
    gate.set()
    warm.join(5)
    first.join(5)
    second.join(5)
    return client, first, second, calls


def test_a_degenerate_output_does_not_wedge_the_offline_client_forever(monkeypatch):
    """A partway batch failure must not kill the only batching thread.

    Resolving outputs in a loop and, on a partway raise, calling
    `future.set_exception` on EVERY future -- including ones already FINISHED --
    raises InvalidStateError inside the handler. That kills the dispatcher,
    leaves every later post() in an undrained queue, and blocks forever on an
    untimed future.result(): the shard hangs, writes no output, and discards
    every row already scored.

    The trigger is real -- a preempted request comes back with `outputs == []`
    -- and it has to land at index >= 1, because a raise on the first output
    leaves nothing already resolved.
    """
    def chat(conversations):
        return [_PreemptedOutput() if c[-1]["content"] == "bad" else _StubOutput()
                for c in conversations]

    good = {"messages": [{"role": "user", "content": "good"}],
            "max_tokens": 4, "temperature": 0}
    bad = {"messages": [{"role": "user", "content": "bad"}],
           "max_tokens": 4, "temperature": 0}
    client, first, second, calls = _batched(monkeypatch, chat, good, bad)

    assert first.response is not None, "the good job in the batch lost its result"
    assert not second.is_alive(), (
        "the caller of the degenerate job never returned — the dispatcher died "
        "inside its own failure handler and nobody will ever resolve its future"
    )
    assert second.error is not None
    assert client.worker.is_alive(), "the batching thread did not survive"

    later = _Caller(client, good)
    later.start()
    later.join(5)
    assert later.response is not None, (
        "a later, perfectly valid request was never served: the client is dead"
    )
    client.close()


def test_one_unprocessable_conversation_fails_only_itself(monkeypatch):
    """One unprocessable conversation must fail only itself.

    `llm.chat()` is all-or-nothing, so one over-length conversation can fail up
    to 96 requests because batch_size defaults to --workers. Retries rebatch the
    offender and multiply the cost by --retries; singleton reissue lands the
    failure on its owning row.
    """
    def chat(conversations):
        if any(c[-1]["content"] == "too long" for c in conversations):
            raise ValueError("prompt is longer than the maximum model length")
        return [_StubOutput() for _ in conversations]

    good = {"messages": [{"role": "user", "content": "fine"}],
            "max_tokens": 4, "temperature": 0}
    bad = {"messages": [{"role": "user", "content": "too long"}],
           "max_tokens": 4, "temperature": 0}
    client, first, second, calls = _batched(monkeypatch, chat, good, bad)

    assert first.response is not None, (
        "a healthy job was failed by its neighbour's over-length prompt"
    )
    assert isinstance(second.error, ValueError)
    client.close()


def test_a_batch_does_not_inherit_the_first_payloads_template_arguments(monkeypatch):
    """Each batch must be homogeneous in chat-template arguments.

    The probe body carries `enable_thinking: False` plus the two continuation
    flags, while a scoring body carries neither. If kwargs come from payloads[0],
    whichever request arrives first inside the 2 ms window decides the thinking
    channel and prefill geometry for up to 96 unrelated jobs. Same input,
    different verdict, decided by thread timing and unrecorded in the output.
    """
    # `len(batched) == 2` alone is equally true of a client that never batches,
    # which is the code path this test exists to rule out. Recording when each
    # payload was TAKEN INTO A BATCH (`_template_signature` runs once per item of
    # the assembled batch) against when it was dispatched separates the two:
    # one batch reads queued,queued,chat,chat; no batching reads
    # queued,chat,queued,chat.
    events: list[tuple[str, str]] = []
    real_signature = runner._template_signature

    def recording_signature(payload):
        events.append(("queued", payload["messages"][-1]["content"]))
        return real_signature(payload)

    monkeypatch.setattr(runner, "_template_signature", recording_signature)

    def chat(conversations):
        events.append(("chat", conversations[0][-1]["content"]))
        return [_StubOutput() for _ in conversations]

    probe = {
        "messages": [{"role": "system", "content": "s"},
                     {"role": "user", "content": "u"},
                     {"role": "assistant", "content": '{"verdict": "'}],
        "max_tokens": 1,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "continue_final_message": True,
        "add_generation_prompt": False,
    }
    scoring = {"messages": [{"role": "user", "content": "score me"}],
               "max_tokens": 64, "temperature": 0}
    client, first, second, calls = _batched(monkeypatch, chat, probe, scoring)
    client.close()

    ordering = [kind for kind, content in events
                if content in {'{"verdict": "', "score me"}]
    assert ordering == ["queued", "queued", "chat", "chat"], (
        "the two payloads never shared a batch, so this test would pass against "
        f"a client that dispatches one request at a time: {ordering}"
    )

    batched = [call for call in calls
               if any(c[-1]["content"] in {'{"verdict": "', "score me"}
                      for c in call["conversations"])]
    assert len(batched) == 2, (
        "the probe and the scoring request were served by one llm.chat call, so "
        "one of them ran under the other's chat template arguments"
    )
    by_content = {call["conversations"][0][-1]["content"]: call["kwargs"]
                  for call in batched}
    assert by_content['{"verdict": "'] == {
        "chat_template_kwargs": {"enable_thinking": False},
        "continue_final_message": True,
        "add_generation_prompt": False,
    }, "the probe's forced-position geometry did not reach llm.chat"
    assert by_content["score me"] == {}, (
        "the scoring call inherited the probe's thinking suppression"
    )


def test_the_probe_request_reaches_the_offline_backend_intact(monkeypatch):
    """The real probe body, not a hand-written one.

    `build_probe_request(..., inline_extra_body=True)` puts
    `continue_final_message` and `add_generation_prompt` at the TOP LEVEL of the
    body. The offline client forwarded only `chat_template_kwargs`, so vLLM ran
    at its defaults (add_generation_prompt=True, continue_final_message=False):
    the assistant prefill was CLOSED and a fresh turn opened, which puts
    generated position 0 somewhere that is not the verdict. The reader indexes
    position 0 regardless, so the run persisted either a margin measured at the
    wrong position or -- more often -- nothing at all, while reporting clean.
    """
    from indra_belief.probes.reader import build_probe_request

    body = build_probe_request(
        {"subject": "A", "object": "B", "stmt_type": "Activation",
         "evidence_text": "A activates B."},
        model_id="served", top_logprobs=128, inline_extra_body=True,
    )
    seen: list = []

    def chat(conversations):
        return [_StubOutput() for _ in conversations]

    _install_vllm(monkeypatch, chat, calls=seen)
    with runner.OfflineVllmClient(
        "model/path", batch_size=2, gpu_memory_utilization=0.8, timeout=5
    ) as client:
        client.post("", json=body)

    assert seen[0]["kwargs"]["continue_final_message"] is True
    assert seen[0]["kwargs"]["add_generation_prompt"] is False
    assert seen[0]["kwargs"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_the_offline_backend_honours_a_json_object_response_format(monkeypatch):
    """The relation-nature call requires JSON; dropping it changes the verdict.

    `score_job` sends `response_format: {"type": "json_object"}` for a [Complex]
    claim. If SamplingParams carries only temperature, max_tokens and logprobs,
    the model answers free text, `_extract_json` finds no
    object, `relation_note` returns "" and the claim is scored WITHOUT the
    rejection note the relnature variant exists for -- differently from the same
    job on --backend server, with no error and no counter.
    """
    class GuidedDecodingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def chat(conversations):
        return [_StubOutput() for _ in conversations]

    _install_vllm(monkeypatch, chat,
                  structured=types.SimpleNamespace(
                      GuidedDecodingParams=GuidedDecodingParams))
    seen: list = []
    with runner.OfflineVllmClient(
        "model/path", batch_size=2, gpu_memory_utilization=0.8, timeout=5
    ) as client:
        assert client.structured_outputs is not None, (
            "the client did not find this vLLM's structured-output parameters"
        )
        params = client._sampling_params({
            "messages": [], "max_tokens": 64, "temperature": 0.1,
            "response_format": {"type": "json_object"},
        })
        seen.append(params)

    guided = seen[0].kwargs.get("guided_decoding")
    assert guided is not None, (
        "response_format was dropped: the offline model answers free text and "
        "the Complex rejection note silently disappears"
    )
    assert guided.kwargs == {"json_object": True}


def test_the_offline_engine_is_told_the_window_the_run_will_ask_for(monkeypatch):
    """vLLM's max_logprobs defaults to 20 and the shipped variant asks for 128.

    Undeclared, the engine rejects every request -- after a 26B model has been
    loaded, and by way of a preflight message about an HTTP server the offline
    backend never contacts.
    """
    from indra_belief.model_client import LOCAL_MODELS

    instances: list = []
    _install_vllm(monkeypatch, lambda conversations: [], instances=instances)
    client = runner.OfflineVllmClient(
        "model/path", batch_size=2, gpu_memory_utilization=0.8,
        max_logprobs=128, timeout=5,
    )
    client.close()
    assert instances[0].kwargs.get("max_logprobs") == 128

    # Each of the three widths must win independently. A single
    # `>= variant.in_call_label_logprobs` assertion is vacuous: the registry's
    # 1024 against a 128-wide variant passes even when two terms are dropped.
    from indra_belief.probes.reader import PROBE_TOP_LOGPROBS

    def _variant(window):
        return types.SimpleNamespace(in_call_label_logprobs=window)

    assert runner.offline_max_logprobs(
        {"max_top_logprobs": 512}, _variant(128), False) == 512, (
        "the server ceiling this entry declares was not mirrored, so a body the "
        "HTTP backend accepts is rejected in-process"
    )
    assert runner.offline_max_logprobs(
        {"max_top_logprobs": 8}, _variant(128), False) == 128, (
        "the window on EVERY request lost to a narrower registry ceiling"
    )
    assert runner.offline_max_logprobs(
        {"max_top_logprobs": 8}, _variant(2), True) == PROBE_TOP_LOGPROBS, (
        "--probe widens on demand and the engine was never told"
    )
    assert runner.offline_max_logprobs({}, _variant(0), False) is None, (
        "an undeclared run must leave the engine at its own default rather than "
        "assert a width nothing asked for"
    )

    variant = runner.MonolithicPrompt(runner.DEFAULT_VARIANT).variant
    assert runner.offline_max_logprobs(
        LOCAL_MODELS["vllm-gemma-4-26b"], variant, False
    ) == 1024


def test_post_refuses_rather_than_parks_when_the_dispatcher_is_gone(monkeypatch):
    """`post` must bound future.result() and refuse when the dispatcher is gone.

    Without a dispatcher, an unbounded wait parks every worker, prevents the
    ThreadPoolExecutor from draining, and hangs a GPU-holding run instead of
    failing one job.
    """
    import concurrent.futures as _cf

    import pytest as _pytest

    def slow(conversations):
        time.sleep(5)
        return [_StubOutput() for _ in conversations]

    _install_vllm(monkeypatch, slow)
    client = runner.OfflineVllmClient(
        "model/path", batch_size=2, gpu_memory_utilization=0.8, timeout=0.2
    )
    started = time.perf_counter()
    with _pytest.raises(_cf.TimeoutError):
        client.post("", json={"messages": [], "max_tokens": 1, "temperature": 0})
    assert time.perf_counter() - started < 4, "the wait was not bounded"

    client.worker.join(0)
    client.requests.put(None)
    client.worker.join(6)
    with _pytest.raises(RuntimeError, match="dispatcher is not running"):
        client.post("", json={"messages": [], "max_tokens": 1, "temperature": 0})


def test_the_documented_serve_flag_covers_the_window_this_run_asks_for():
    """The runbook's `vllm serve` line is the only record of the server's cap.

    Every request the default variant sends carries a logprob window, and vLLM
    rejects the WHOLE call -- verdict included -- when it exceeds
    `--max-logprobs`. So a recipe narrower than the variant's window fails every
    row of a 60M run for a flag, and the registry's own 1024 (the direct
    probe's ceiling, its losing label measured at rank 42/83/168) is an upper
    bound on what a caller may ask for rather than a record of how that server
    was started.
    """
    import re

    from indra_belief.scorers.monolithic import scorer as mono

    runbook = (Path(__file__).resolve().parents[1]
               / "research" / "corpus_belief_runbook.md").read_text()
    documented = [int(value) for value in
                  re.findall(r"vllm serve[^\n]*--max-logprobs (\d+)", runbook)]
    assert documented, "the runbook no longer documents how to start the server"
    window = mono.VARIANTS[runner.DEFAULT_VARIANT].in_call_label_logprobs
    for value in documented:
        assert value >= window, (
            f"the runbook starts the server at --max-logprobs {value} while "
            f"{runner.DEFAULT_VARIANT} asks for {window} on every request"
        )


def test_preflight_blames_the_engine_not_the_base_url_on_the_offline_backend():
    """The offline backend contacts no endpoint, so `--base-url` is a red herring.

    The operator has just spent minutes loading a 26B model; sending them to
    check whether an HTTP server is up diagnoses something that was never
    involved.
    """
    import pytest as _pytest

    class Dead:
        # THE REAL ATTRIBUTE. A locally declared string makes this pass against
        # its own stub even when OfflineVllmClient leaves preflight blaming
        # --base-url.
        transport_description = runner.OfflineVllmClient.transport_description

        def post(self, *_args, **_kwargs):
            raise ValueError("Requested sample logprobs of 128, greater than 20")

    with _pytest.raises(SystemExit) as excinfo:
        runner.preflight(Dead(), "http://127.0.0.1:8000/v1/chat/completions",
                         "served", runner.MonolithicPrompt(runner.DEFAULT_VARIANT))
    message = str(excinfo.value)
    assert "in-process vLLM engine" in message
    assert "is the server up" not in message, (
        "an offline failure still points at --base-url"
    )


# ── retries, reconciliation, publication ──────────────────────────────────────


def test_an_unparseable_reply_is_retried_because_temperature_zero_is_not_a_replay():
    """Temperature 0 does not make identical request bytes a replay.

    REJECTED PREMISE: "the variant pins temperature 0 and the request bytes are
    identical on every attempt, so this reply is this reply". A
    continuously-batched vLLM server is not bitwise reproducible at temperature
    0 -- the floating-point
    reduction order follows the batch composition, and gemma-4-26B-A4B is an MoE
    whose expert routing varies with batch shape. The identical bytes reissued
    into a different batch genuinely can come back parseable, so a reply the
    parser could not read must keep its retries.
    """
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "I am sorry, I cannot."},
                                 "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 9}}

    class Client:
        def __init__(self):
            self.posts = 0

        def post(self, *_args, **_kwargs):
            self.posts += 1
            return Response()

    client = Client()
    row = runner.score_job_with_retries(
        {"job_id": "1:0", "stmt_hash": 1, "source_hash": 1, "needs_llm": True,
         "stmt_type": "Activation", "user_message": "CLAIM: A [Activation] B"},
        retries=3,
        client=client,
        prompt=runner.MonolithicPrompt(runner.DEFAULT_VARIANT),
        endpoint="unused",
        model_id="served",
        max_tokens=64,
        temperature=0.0,
    )

    assert row["error"] == "unparseable model response"
    assert client.posts == 4, (
        f"an unreadable reply was given up on after {client.posts} attempt(s); "
        "batched serving can return a parseable reply to the same bytes"
    )
    assert row["attempts"] == 4, "attempts records the plan, not what was asked"
    assert row["error_class"] == "deterministic", (
        "four attempts returned the IDENTICAL refusal and that was classed "
        "transient — so the publication gate's refusal tells the operator to "
        "rerun, and the rerun reproduces the same four replies at the "
        "registry's 8192-token ceiling, forever"
    )


def test_an_unreadable_reply_that_changes_between_attempts_stays_transient():
    """The other half, and the reason the retries exist at all.

    Batch composition decides the reduction order, so a reply garbled by one
    batch is a different reply in the next. That row's failure is a window, not
    a property of its prompt bytes, and the publication gate must still count it
    — otherwise a real outage publishes as missing evidence.
    """
    class Client:
        def __init__(self):
            self.posts = 0

        def post(self, *_args, **_kwargs):
            self.posts += 1
            return types.SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {
                    "choices": [{"message": {"content": f"<garbled {self.posts}>"},
                                 "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 9}},
            )

    client = Client()
    row = runner.score_job_with_retries(
        {"job_id": "1:0", "stmt_hash": 1, "source_hash": 1, "needs_llm": True,
         "stmt_type": "Activation", "user_message": "CLAIM: A [Activation] B"},
        retries=3,
        client=client,
        prompt=runner.MonolithicPrompt(runner.DEFAULT_VARIANT),
        endpoint="unused",
        model_id="served",
        max_tokens=64,
        temperature=0.0,
    )

    assert row["error"] == "unparseable model response"
    assert row["error_class"] == "transient"


def _refusing_client(first_attempt_raises=None, tail=""):
    """A client that returns one fixed unreadable reply, after an optional fault."""
    class Client:
        def __init__(self):
            self.posts = 0

        def post(self, *_args, **_kwargs):
            self.posts += 1
            if self.posts == 1 and first_attempt_raises is not None:
                raise first_attempt_raises
            content = "I am sorry, I cannot help with that." + tail(self.posts)
            return types.SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {
                    "choices": [{"message": {"content": content},
                                 "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 9}},
            )

    return Client()


def _score_one(client, retries=3):
    return runner.score_job_with_retries(
        {"job_id": "1:0", "stmt_hash": 1, "source_hash": 1, "needs_llm": True,
         "stmt_type": "Activation", "user_message": "CLAIM: A [Activation] B"},
        retries=retries,
        client=client,
        prompt=runner.MonolithicPrompt(runner.DEFAULT_VARIANT),
        endpoint="unused",
        model_id="served",
        max_tokens=64,
        temperature=0.0,
    )


def test_one_transport_failure_among_identical_refusals_stays_transient():
    """A MIXED sequence is not evidence about the prompt bytes.

    A transport failure has no reply to compare, so it contributes None and the
    `None not in replies` guard is what keeps "one timeout, three identical
    refusals" out of the deterministic class. Without it that row is reported to
    the operator as unclearable -- so the refusal names the wrong remedy, and
    the rerun that would in fact have cleared the timed-out attempt is the one
    thing the message argues against.
    """
    client = _refusing_client(first_attempt_raises=ConnectionError("reset"),
                              tail=lambda n: "")
    row = _score_one(client)

    assert client.posts == 4 and row["attempts"] == 4
    assert row["error_class"] == "transient", (
        "a row whose first attempt never reached the model was reported as a "
        "permanent property of its prompt"
    )


def test_the_deterministic_class_is_decided_on_a_PREFIX_of_the_reply():
    """What the signature establishes, which is less than "the same text".

    `response_preview` is truncated to 4,000 characters as a diagnostic cap, so
    two attempts with the same signature agree on their first 4,000 characters
    and their finish reason -- not on the whole reply. That is the evidence the
    deterministic class actually rests on, and a docstring claiming the replies
    were identical overstates it: a degenerate repetition that diverges at
    character 4,001 lands in the class regardless.
    """
    client = _refusing_client(tail=lambda n: " " + "z" * 4000 + f" tail {n}")
    row = _score_one(client)

    assert row["error"] == runner.UNPARSEABLE_REPLY
    assert client.posts == 4
    assert len(row["response_preview"]) == 4000, (
        "the preview is no longer the truncated one the class is decided on"
    )
    assert row["error_class"] == "deterministic", (
        "four replies agreeing on their first 4,000 characters were split "
        "apart by bytes nothing ever compares"
    )


def test_a_job_with_no_user_message_is_terminal_on_the_first_attempt():
    """The other side: a failure that IS a property of the input row.

    Building the request raises before any request is sent, and it raises the
    same way three more times. Nothing here reaches the model, so `error_class`
    must say deterministic. The gate counts BOTH classes, so the class does not
    decide whether the shard is withheld -- it decides what the refusal TELLS
    the operator, and a row no rerun can clear must not be reported as one that
    a rerun would.
    """
    class Client:
        def __init__(self):
            self.posts = 0

        def post(self, *_args, **_kwargs):
            self.posts += 1
            raise AssertionError("a job with no user_message reached the model")

    client = Client()
    row = runner.score_job_with_retries(
        {"job_id": "1:0", "stmt_hash": 1, "source_hash": 1, "needs_llm": True,
         "stmt_type": "Activation", "user_message": "   "},
        retries=3,
        client=client,
        prompt=runner.MonolithicPrompt(runner.DEFAULT_VARIANT),
        endpoint="unused",
        model_id="served",
        max_tokens=64,
        temperature=0.0,
    )
    assert client.posts == 0
    assert row["attempts"] == 1
    assert row["error_class"] == "deterministic"


def test_a_transport_failure_is_still_retried():
    """The other half: a failure a rerun COULD change must keep its retries."""
    class Client:
        def __init__(self):
            self.posts = 0

        def post(self, *_args, **_kwargs):
            self.posts += 1
            raise ConnectionError("connection refused")

    client = Client()
    row = runner.score_job_with_retries(
        {"job_id": "1:0", "stmt_hash": 1, "source_hash": 1, "needs_llm": True,
         "stmt_type": "Activation", "user_message": "CLAIM: A [Activation] B"},
        retries=3,
        client=client,
        prompt=runner.MonolithicPrompt(runner.DEFAULT_VARIANT),
        endpoint="unused",
        model_id="served",
        max_tokens=64,
        temperature=0.0,
    )
    assert client.posts == 4
    assert row["attempts"] == 4
    assert row["error_class"] == "transient"


def test_a_four_hundred_is_terminal_because_the_bytes_do_not_change():
    class _Refusal:
        status_code = 400
        text = "prompt is longer than the maximum model length"

    class Client:
        def __init__(self):
            self.posts = 0

        def post(self, *_args, **_kwargs):
            self.posts += 1
            error = RuntimeError("400 Bad Request: context length exceeded")
            error.response = _Refusal()
            raise error

    client = Client()
    row = runner.score_job_with_retries(
        {"job_id": "1:0", "stmt_hash": 1, "source_hash": 1, "needs_llm": True,
         "stmt_type": "Activation", "user_message": "CLAIM: A [Activation] B"},
        retries=3,
        client=client,
        prompt=runner.MonolithicPrompt(runner.DEFAULT_VARIANT),
        endpoint="unused",
        model_id="served",
        max_tokens=64,
        temperature=0.0,
    )
    assert client.posts == 1, "an over-length prompt paid four full generations"
    assert row["error_class"] == "deterministic", (
        "a 4xx is the server rejecting the request BYTES; no rerun clears "
        "it, so the refusal must not offer a rerun as the remedy"
    )


def test_a_malformed_tier1_verdict_is_terminal_too():
    """The tier1 verdict is DATA on the input job, not an answer from a server.

    Rereading it produces the same malformed dict every time, so it belongs with
    the failures no rerun can clear: a shard of rows the
    grounding stage wrote badly would otherwise be withheld by every rerun.
    """
    class Client:
        def post(self, *_args, **_kwargs):
            raise AssertionError("a tier1 job reached the model")

    row = runner.score_job_with_retries(
        {"job_id": "1:0", "stmt_hash": 1, "source_hash": 1, "needs_llm": False,
         "tier1_result": {"verdict": "maybe", "confidence": "high"}},
        retries=3,
        client=Client(),
        prompt=runner.MonolithicPrompt(runner.DEFAULT_VARIANT),
        endpoint="unused",
        model_id="served",
        max_tokens=64,
        temperature=0.0,
    )
    assert row["error"] == "bad tier1"
    assert row["error_class"] == "deterministic"


def test_a_repeated_pair_does_not_let_an_error_erase_a_scored_verdict(tmp_path):
    """Repeated jobs for one pair must be reconciled independently of file order.

    The run is keyed on job_id; the published file is keyed on the PAIR.
    Under --all-evidence a statement can carry the same evidence twice, which is
    two jobs and one cell. Direct assignment lets the later job win by file
    order, so an errored duplicate erases a good verdict and its margin -- the
    belief build then counts BOTH jobs as unscored and the statement disappears
    from the table. Reverse order masks the error instead. Reconcile with the
    same rule statement_belief applies to repeated evidence, and count the
    duplicate so a shard cannot lose reads in silence.
    """
    shard = tmp_path / "grounded-000000.jsonl.gz"
    write_jobs(
        shard,
        [
            {"job_id": "5:0", "stmt_hash": 111, "source_hash": 77},
            {"job_id": "5:1", "stmt_hash": 111, "source_hash": 77},
        ],
    )
    latest = {
        "5:0": {"verdict": "correct", "confidence": "medium",
                "probe_delta_logit": 3.2},
        "5:1": {"verdict": None, "confidence": None, "error": "ReadTimeout",
                "attempts": 4},
    }
    stats: dict = {}

    payload = runner.finalize(shard, latest, None, stats=stats)

    assert payload["111"]["77"]["verdict"] == "correct", (
        "an exhausted duplicate overwrote the scored verdict"
    )
    assert payload["111"]["77"]["probe_delta_logit"] == 3.2
    assert stats["duplicate_pairs"] == 1


def test_a_repeated_pair_reconciles_conservatively(tmp_path):
    """Two readable verdicts for one pair: any-incorrect-wins, not file order."""
    shard = tmp_path / "grounded-000000.jsonl.gz"
    write_jobs(
        shard,
        [
            {"job_id": "5:0", "stmt_hash": 111, "source_hash": 77},
            {"job_id": "5:1", "stmt_hash": 111, "source_hash": 77},
        ],
    )
    latest = {
        "5:0": {"verdict": "incorrect", "confidence": "high"},
        "5:1": {"verdict": "correct", "confidence": "high"},
    }
    payload = runner.finalize(shard, latest, None)
    assert payload["111"]["77"]["verdict"] == "incorrect"


def test_the_sole_candidate_for_a_shard_is_not_a_different_generation(tmp_path):
    """A limited smoke-test output must not stand in for the full shard.

    `--shard-index 0 --limit 1000` is this file's own documented example. Its
    output can be the only candidate for shard 0; a glob fallback then returns it
    to a caller asking for the unlimited shard. The remaining 49,000 evidences
    fall to n_unscored, every statement past the limit vanishes from the belief
    table, and the manifest looks healthy because a shard is read.
    `allow_limited` is the explicit opt-in for a caller that accepts that
    generation.
    """
    (tmp_path / "verdicts-000000.limit-1000.json.gz").write_bytes(b"")

    assert runner.resolve_results_path(tmp_path, 0, None) is None
    assert runner.resolve_results_path(tmp_path, 0, 999) is None
    assert runner.resolve_results_path(
        tmp_path, 0, None, allow_limited=True
    ).name == "verdicts-000000.limit-1000.json.gz"


def test_write_final_atomic_fsyncs_the_payload_before_publishing_it(tmp_path,
                                                                    monkeypatch):
    """The rename is metadata; the 9MB payload is data.

    `gzip.open` + `replace` survives SIGKILL (the page cache outlives the
    process) but not a node crash: the journal can commit the rename while the
    data extents are still dirty, leaving a zero-length or NUL-filled file that
    the completion check reads as a finished shard forever.
    """
    events: list[str] = []
    real_fsync = runner.os.fsync
    real_replace = Path.replace

    def record_fsync(fd):
        events.append("fsync")
        return real_fsync(fd)

    def record_replace(self, target):
        events.append("rename")
        return real_replace(self, target)

    monkeypatch.setattr(runner.os, "fsync", record_fsync)
    monkeypatch.setattr(Path, "replace", record_replace)
    runner.write_final_atomic(tmp_path / "verdicts-000000.json.gz",
                              {"1": {"2": {"verdict": "correct"}}})

    assert events[:2] == ["fsync", "rename"], (
        f"the payload was published before it was durable: {events}"
    )
    assert events.count("fsync") >= 2, (
        "the directory entry itself was never fsynced, so the rename can be lost"
    )


def test_two_writers_of_one_shard_cannot_interleave_into_one_file(tmp_path,
                                                                  monkeypatch):
    """Concurrent writers must use process-unique staging names.

    `.{final}.tmp` is derived only from the shard, so it is SHARED.
    Per-shard parallelism across nodes is the documented usage, and a supervisor
    relaunching a run while its predecessor is still draining is the concrete
    case. The second process can open the same staging inode with O_TRUNC while
    the first is mid-write, publishing NULs followed by another stream's tail;
    the loser's rename raises FileNotFoundError.
    """
    final = tmp_path / "verdicts-000800.json.gz"
    both_open = threading.Barrier(2, timeout=10)
    real_gzipfile = gzip.GzipFile
    real_open = gzip.open

    def staged_gzipfile(*args, **kwargs):
        handle = real_gzipfile(*args, **kwargs)
        both_open.wait()
        return handle

    def staged_open(*args, **kwargs):
        handle = real_open(*args, **kwargs)
        both_open.wait()
        return handle

    monkeypatch.setattr(gzip, "GzipFile", staged_gzipfile)
    monkeypatch.setattr(gzip, "open", staged_open)

    failures: list[BaseException] = []

    def writer(payload):
        try:
            runner.write_final_atomic(final, payload)
        except BaseException as exc:  # noqa: BLE001 - the outcome under test
            failures.append(exc)

    threads = [
        threading.Thread(target=writer, args=({"1": {"1": {"verdict": "correct"}}},)),
        threading.Thread(target=writer, args=({"2": {"2": {"verdict": "incorrect"}}},)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(15)

    monkeypatch.undo()
    assert not failures, f"a concurrent writer died: {failures[0]!r}"
    assert runner.published_output_is_readable(final), (
        "two writers sharing one staging name published a corrupt gzip"
    )
    assert not list(tmp_path.glob(".*tmp*")), "a staging file was left behind"


# ── run_shard: what counts as a finished shard ────────────────────────────────


class _Reply:
    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": "verdict correct\nconfidence high"},
                             "finish_reason": "stop"}],
                "usage": {"completion_tokens": 4}}


class _ScoringClient:
    """Answers every scoring request, except the ones named in `fail_for`."""

    def __init__(self, fail_for=()):
        self.fail_for = tuple(fail_for)
        self.posts: list[dict] = []

    def post(self, _endpoint, json, timeout=None):
        self.posts.append(json)
        content = json["messages"][-1]["content"]
        if any(token in content for token in self.fail_for):
            raise ConnectionError("connection refused")
        return _Reply()


def _shard_jobs(count, prefix="A"):
    return [
        {"job_id": f"{i}:0", "input_row_index": i, "stmt_hash": 100 + i,
         "source_hash": 10 + i, "needs_llm": True, "stmt_type": "Activation",
         "user_message": f"CLAIM: {prefix}{i} [Activation] B\n"
                         f"EVIDENCE: {prefix}{i} activates B."}
        for i in range(count)
    ]


def _shard_args(tmp_path, **overrides):
    args = types.SimpleNamespace(
        output_dir=str(tmp_path / "out"),
        limit=None,
        workers=2,
        retries=0,
        probe=False,
        model="vllm-gemma-4-26b",
        model_id="served-model",
        max_tokens=64,
        temperature=0.0,
        base_url="http://vllm/v1",
        max_error_fraction=0.01,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _score_shard(tmp_path, jobs, client, stmt_hashes=None, **overrides):
    shard = tmp_path / "grounded-000007.jsonl.gz"
    if not shard.exists():
        write_jobs(shard, jobs)
    args = _shard_args(tmp_path, **overrides)
    prompt = runner.MonolithicPrompt(getattr(args, "variant", runner.DEFAULT_VARIANT))
    outcome = runner.run_shard(shard, args, client, prompt, stmt_hashes)
    return outcome, runner.output_path(Path(args.output_dir), 7, args.limit)


def test_a_published_shard_that_does_not_read_back_is_rescored(tmp_path):
    """`final_path.exists()` cannot tell a finished shard from wreckage.

    A crash between the rename and writeback leaves a zero-length or NUL-filled
    gzip. A skip based only on existence treats it as complete forever, so the
    50,000 verdicts are never regenerated and the corruption surfaces one stage
    later as a BadGzipFile -- with the corrupt-file-to-shard correlation left to
    a human.
    """
    jobs = _shard_jobs(3)
    outcome, final = _score_shard(tmp_path, jobs, _ScoringClient())
    assert outcome.code == 0 and final.exists()

    final.write_bytes(b"")
    second = _ScoringClient()
    outcome, final = _score_shard(tmp_path, jobs, second)

    assert outcome.code == 0
    assert second.posts, "the corrupt output was skipped as a completed shard"
    assert runner.published_output_is_readable(final)


def test_a_window_of_failures_is_not_sealed_into_the_corpus(tmp_path):
    """A transient outage must not become the shard's final answer.

    Every job in flight during a vLLM restart exhausts its attempts in
    milliseconds and finalizes as verdict="error". Publishing those rows seals
    the outage:
    the next run skips a completed shard, and downstream those evidences are
    dropped as unscored, so their statements are published with a belief
    computed from a fraction of their reads -- a wrong number, not an absent
    one. The scored rows must survive the refusal, so the rerun pays only for
    the failures.
    """
    jobs = _shard_jobs(5)
    outage = _ScoringClient(fail_for=("A3 ", "A4 "))
    outcome, final = _score_shard(tmp_path, jobs, outage)

    assert outcome.code == 2, "a shard that failed 40% of its jobs was published"
    assert "TRANSIENT" in outcome.reason and "shard 7" in outcome.reason
    assert not final.exists()

    healthy = _ScoringClient()
    outcome, final = _score_shard(tmp_path, jobs, healthy)
    assert outcome.code == 0
    assert len(healthy.posts) == 2, (
        f"the rerun rescored {len(healthy.posts)} jobs; the three that succeeded "
        "before the outage were discarded with the shard"
    )
    with gzip.open(final, "rt") as fh:
        published = json.load(fh)
    assert not [cell for by_source in published.values()
                for cell in by_source.values() if cell["verdict"] == "error"]
    assert not list(Path(tmp_path / "out").glob(".*partial.jsonl"))


def test_a_gene_filtered_shard_does_not_satisfy_an_unfiltered_rerun(tmp_path):
    """The output NAME carries --limit but cannot carry --gene-stmt-hashes.

    With an explicit --output-dir (the norm, since the defaults are one
    machine's /scratch paths), a filtered run and an unfiltered run collide on
    verdicts-NNNNNN.json.gz. The generic name can mark every shard "complete",
    print skip for all of them, and exit 0 having scored nothing; the belief
    build then publishes only the gene subset under a manifest that says nothing
    about a filter.
    """
    jobs = _shard_jobs(3)
    outcome, final = _score_shard(tmp_path, jobs, _ScoringClient(), stmt_hashes={101})
    assert outcome.code == 0
    with gzip.open(final, "rt") as fh:
        assert set(json.load(fh)) == {"101"}

    unfiltered = _ScoringClient()
    outcome, _ = _score_shard(tmp_path, jobs, unfiltered)
    assert outcome.code == 2 and "stmt_hash_filter" in outcome.reason
    assert not unfiltered.posts, "the filtered output was rescored over"


def test_a_disagreeing_sidecar_withholds_ITS_shard_and_not_the_run(tmp_path,
                                                                   monkeypatch,
                                                                   capsys):
    """A per-shard provenance conflict must not halt the shard loop.

    Raising SystemExit from `provenance_conflict` inside `run_shard` stops every
    shard queued behind one disagreeing sidecar, discarding hours of work over a
    per-shard configuration mismatch. Sidecar-less shards cannot conflict; the
    check applies once a rescored shard writes one and another run changes
    --served-model-id or --variant.
    """
    import sys

    inputs = tmp_path / "shards"
    inputs.mkdir()
    write_jobs(inputs / "grounded-000000.jsonl.gz", _shard_jobs(2, prefix="A"))
    write_jobs(inputs / "grounded-000001.jsonl.gz", _shard_jobs(2, prefix="B"))
    out = tmp_path / "out"
    out.mkdir()
    # shard 0 carries a different served id from the active run
    runner.write_final_atomic(out / "verdicts-000000.json.gz",
                              {"100": {"10": {"verdict": "correct"}}})
    runner.write_shard_meta(out / "verdicts-000000.meta.json",
                            {"served_model_id": "some-other-model"})

    client = _ScoringClient()

    class _Engine:
        transport_description = runner.OfflineVllmClient.transport_description

        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return client

        def __exit__(self, *_exc):
            return None

    monkeypatch.setattr(runner, "OfflineVllmClient", _Engine)
    monkeypatch.setattr(runner, "preflight", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", [
        "run_vllm_processed_shards.py",
        "--input-dir", str(inputs), "--output-dir", str(out),
        "--model", "vllm-local", "--backend", "offline",
        "--workers", "2", "--retries", "0",
    ])

    code = runner.main()
    printed = capsys.readouterr().out

    assert code == 2
    assert (out / "verdicts-000001.json.gz").exists(), (
        "shard 1 never ran; one disagreeing sidecar halted the shards behind it"
    )
    assert "served_model_id" in printed and "1 of 2 shards were NOT published" in printed


def test_a_scored_shard_records_the_configuration_that_produced_it(tmp_path):
    """The sidecar must identify the scorer configuration.

    The published dict cannot say who scored it. Model, served id, variant,
    prompt sha and the acquisition route of
    `probe_delta_logit` are all absent from {stmt_hash: {source_hash: cell}}, so
    the belief build's --model/--variant are unverifiable assertions that can
    select a profile silently and publish the assertion as fact in its manifest.
    """
    outcome, final = _score_shard(tmp_path, _shard_jobs(2), _ScoringClient())
    assert outcome.code == 0
    meta = json.loads(runner.meta_path_for(final).read_text())
    assert meta["model"] == "vllm-gemma-4-26b"
    assert meta["served_model_id"] == "served-model"
    assert meta["variant"] == runner.DEFAULT_VARIANT
    assert len(meta["prompt_sha256"]) == 64
    assert meta["margin_route"] == "pol.verdict_incall"
    assert meta["n_jobs"] == 2 and meta["n_errors"] == 0


# ── the publication gate, and what it may count ───────────────────────────────


def test_a_shard_of_permanently_failing_rows_needs_a_deliberate_operator_act(
    tmp_path, capsys
):
    """A shard of pure deterministic failure must not publish itself.

    Gating only TRANSIENT failures lets a shard whose every cell is
    verdict="error" exit 0 and publish itself: a systematically broken prompt or
    input bakes into the corpus silently. Both classes count; the escape from the
    otherwise unclearable wedge is the operator knowingly raising
    --max-error-fraction, an explicit human decision rather than an unreachable
    state.

    The two classes still have DIFFERENT remedies, so the refusal names the
    split: transient failures clear on a rerun, deterministic ones never will.
    """
    jobs = _shard_jobs(5)
    for job in jobs[:2]:
        job["user_message"] = "   "

    outcome, final = _score_shard(tmp_path, jobs, _ScoringClient())

    assert outcome.code == 2, "40% deterministic failures published silently"
    assert not final.exists()
    assert "DETERMINISTIC 2" in outcome.reason and "TRANSIENT 0" in outcome.reason
    refusal = capsys.readouterr().out
    # Pin the DISTINCT half of each remedy. Generic "rerun" and
    # "--max-error-fraction" assertions are vacuous because both words occur
    # elsewhere in the refusal.
    assert "TRANSIENT ones, which can come back different" in refusal, (
        "the refusal no longer says what a rerun is worth doing FOR"
    )
    assert "bake them into the corpus knowingly" in refusal, (
        "the refusal no longer names publishing-anyway as a deliberate act"
    )
    assert "NOT for the 3 that succeeded" in refusal, (
        "the refusal no longer states what a rerun re-pays for; it once "
        "claimed a rerun paid only for the transient rows, which is false -- "
        "resume skips a row only when its result was valid, so BOTH classes "
        "are rescored"
    )

    # THE PRINTED RATE MUST ACTUALLY CLEAR. A truncated rate is worse than
    # none: an operator who follows it sets a threshold the rate still exceeds
    # and gets the identical refusal back.
    printed = re.search(r"--max-error-fraction ([0-9.]+) or higher", refusal)
    assert printed, refusal
    outcome, final = _score_shard(tmp_path, jobs, _ScoringClient(),
                                  max_error_fraction=float(printed.group(1)))
    assert outcome.code == 0, (
        f"the rate the refusal printed did not clear it: {outcome.reason}"
    )

    # The escape hatch: the operator, having read the split, publishes anyway.
    outcome, final = _score_shard(tmp_path, jobs, _ScoringClient(),
                                  max_error_fraction=0.4)
    assert outcome.code == 0, (
        f"raising the bar did not publish the shard: {outcome.reason}"
    )
    meta = json.loads(runner.meta_path_for(final).read_text())
    assert meta["n_errors"] == 2
    assert meta["n_errors_deterministic"] == 2, (
        "a genuinely broken prompt would be invisible in the sidecar"
    )
    assert meta["n_errors_transient"] == 0
    with gzip.open(final, "rt") as fh:
        published = json.load(fh)
    # The split lives in the sidecar, not the cell: the shard format is consumed
    # by readers that must keep working unchanged, and the ~1,200 shards already
    # published by the live run have no such field to be compared against.
    assert {key for by_source in published.values()
            for cell in by_source.values() if cell["verdict"] == "error"
            for key in cell} == {"verdict", "confidence", "error", "attempts"}


def test_the_gate_measures_errors_against_the_rows_it_actually_publishes(
    tmp_path, capsys
):
    """Numerator and denominator must share a basis.

    The gate must derive numerator and denominator from the published PAIR basis.
    Error counts come from `finalize` after duplicates collapse; dividing them by
    the larger JOB count under --all-evidence measures against rows absent from
    the published table. The gate then reads low by exactly the duplicate count:
    here 1/4 = 25% against the true 1/3 = 33%, which straddles a 30% threshold
    and publishes a shard that should be withheld.
    """
    jobs = [
        {"job_id": "0:0", "input_row_index": 0, "stmt_hash": 100,
         "source_hash": 10, "needs_llm": True, "stmt_type": "Activation",
         "user_message": "CLAIM: A [Activation] B\nEVIDENCE: A activates B."},
        # The duplicate pair: a second job, the same evidence on the same
        # statement, which finalize collapses into one published cell.
        {"job_id": "0:1", "input_row_index": 1, "stmt_hash": 100,
         "source_hash": 10, "needs_llm": True, "stmt_type": "Activation",
         "user_message": "CLAIM: A [Activation] B\nEVIDENCE: A activates B."},
        {"job_id": "1:0", "input_row_index": 2, "stmt_hash": 101,
         "source_hash": 11, "needs_llm": True, "stmt_type": "Activation",
         "user_message": "CLAIM: C [Activation] B\nEVIDENCE: C activates B."},
        # The only failure, and it is a whole pair of the three published.
        {"job_id": "2:0", "input_row_index": 3, "stmt_hash": 102,
         "source_hash": 12, "needs_llm": True, "stmt_type": "Activation",
         "user_message": "   "},
    ]

    outcome, final = _score_shard(tmp_path, jobs, _ScoringClient(),
                                  max_error_fraction=0.30)

    assert outcome.code == 2, (
        "1 failure among 3 PUBLISHED pairs is 33%, above the 30% bar; measured "
        "against the 4 JOBS it reads as 25% and the shard publishes"
    )
    assert not final.exists()
    assert "/3 evidence pairs" in outcome.reason, (
        f"the gate is still counting jobs, not published pairs: {outcome.reason}"
    )

    # 1/3 is 0.3333... -- the case that separates a rounded-UP clearing rate
    # from a truncated one. Truncating prints 0.3333, which 0.33333... still
    # exceeds, so an operator who follows the printed advice gets the identical
    # refusal back. The other gate fixture cannot catch this: its rate is
    # exactly 0.4, where truncation and rounding agree.
    printed = re.search(r"--max-error-fraction ([0-9.]+) or higher",
                        capsys.readouterr().out)
    assert printed, "the refusal did not print a rate to clear it"
    cleared, _final = _score_shard(tmp_path, jobs, _ScoringClient(),
                                   max_error_fraction=float(printed.group(1)))
    assert cleared.code == 0, (
        f"the printed rate {printed.group(1)} did not clear a rate of "
        f"1/3: {cleared.reason}"
    )


def test_permanently_UNREADABLE_REPLIES_are_classed_but_still_gated(tmp_path):
    """The class this row lands in decides which remedy the refusal offers.

    An unparseable reply is retried -- batched vLLM at temperature 0 is not
    bitwise reproducible -- but a refusal or a degenerate repetition comes back
    IDENTICAL every time, so no rerun clears it and telling the operator to
    rerun would burn retries+1 generations per bad row at the registry's
    8192-token ceiling, forever. Classifying it is what makes the refusal
    useful; it does not exempt it from the gate, because a shard of nothing but
    refusals is the corpus-poisoning case. Publishing it is the operator's call.
    """
    jobs = _shard_jobs(10)

    class _Refusing(_ScoringClient):
        def post(self, endpoint, json, timeout=None):
            content = json["messages"][-1]["content"]
            if any(f"A{i} " in content for i in (3, 4, 5, 6)):
                self.posts.append(json)
                return types.SimpleNamespace(
                    raise_for_status=lambda: None,
                    json=lambda: {
                        "choices": [{"message": {
                            "content": "I am sorry, I cannot help with that."},
                            "finish_reason": "stop"}],
                        "usage": {"completion_tokens": 9}},
                )
            return super().post(endpoint, json, timeout)

    outcome, final = _score_shard(tmp_path, jobs, _Refusing(), retries=3)
    assert outcome.code == 2 and not final.exists()
    assert "DETERMINISTIC 4" in outcome.reason, (
        f"the refusal blamed a class no rerun can clear on the wrong remedy: "
        f"{outcome.reason}"
    )

    outcome, final = _score_shard(tmp_path, jobs, _Refusing(), retries=3,
                                  max_error_fraction=0.4)
    assert outcome.code == 0, (
        f"the operator raised the bar and the shard still did not publish: "
        f"{outcome.reason}"
    )
    meta = json.loads(runner.meta_path_for(final).read_text())
    assert meta["n_errors_deterministic"] == 4
    assert meta["n_errors_transient"] == 0

    rerun = _Refusing()
    outcome, _ = _score_shard(tmp_path, jobs, rerun, retries=3,
                              max_error_fraction=0.4)
    assert outcome.code == 0 and not rerun.posts, (
        "the rerun paid for the same four rows again; a published shard is skipped"
    )


def test_a_withheld_shard_does_not_halt_the_shards_behind_it(tmp_path, monkeypatch,
                                                             capsys):
    """One withheld shard must not discard 1,199 shards of progress.

    An early return after `run_shard` lets one withheld shard stop every shard
    queued behind it, with only a process exit code after the console line
    scrolls away. Every shard must be attempted, the withheld ones named at the
    end, and the run must exit non-zero ONCE.
    """
    import sys

    import pytest as _pytest

    inputs = tmp_path / "shards"
    inputs.mkdir()
    write_jobs(inputs / "grounded-000000.jsonl.gz", _shard_jobs(5, prefix="A"))
    write_jobs(inputs / "grounded-000001.jsonl.gz", _shard_jobs(5, prefix="B"))

    outage = _ScoringClient(fail_for=("A3 ", "A4 "))

    class _Engine:
        transport_description = runner.OfflineVllmClient.transport_description

        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return outage

        def __exit__(self, *_exc):
            return None

    monkeypatch.setattr(runner, "OfflineVllmClient", _Engine)
    monkeypatch.setattr(runner, "preflight", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", [
        "run_vllm_processed_shards.py",
        "--input-dir", str(inputs),
        "--output-dir", str(tmp_path / "out"),
        "--model", "vllm-local",
        "--backend", "offline",
        "--workers", "2",
        "--retries", "0",
    ])

    code = runner.main()
    out = capsys.readouterr().out

    assert code == 2, "a run that withheld a shard reported success"
    assert not runner.output_path(tmp_path / "out", 0, None).exists()
    assert runner.output_path(tmp_path / "out", 1, None).exists(), (
        "the healthy shard behind the withheld one was never attempted"
    )
    assert "1 of 2 shards were NOT published" in out
    assert "shard 0" in out and "TRANSIENT" in out
    assert _pytest  # the import is the marker that this drives main(), not run_shard


def test_main_declares_the_logprob_window_and_the_timeout_to_the_engine(tmp_path,
                                                                       monkeypatch):
    """main() must pass max_logprobs and timeout to OfflineVllmClient.

    Omitting `max_logprobs=` makes the engine fall back to vLLM's default of 20
    and reject every 128-wide request after loading a 26B model. Omitting
    `timeout=` leaves the unbounded `future.result()` that parks every worker.
    No suite total is quoted here: it drifts with every test added, and a stale
    total reads as a measurement.
    """
    import sys

    import pytest as _pytest

    from indra_belief.model_client import LOCAL_MODELS

    inputs = tmp_path / "shards"
    inputs.mkdir()
    write_jobs(inputs / "grounded-000000.jsonl.gz", _shard_jobs(1))
    recorded: dict = {}

    class _Engine:
        transport_description = runner.OfflineVllmClient.transport_description

        def __init__(self, model_id, **kwargs):
            recorded["model_id"] = model_id
            recorded.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def post(self, *_args, **_kwargs):
            raise ConnectionError("the engine refused the first request")

    monkeypatch.setattr(runner, "OfflineVllmClient", _Engine)
    monkeypatch.setattr(sys, "argv", [
        "run_vllm_processed_shards.py",
        "--input-dir", str(inputs),
        "--output-dir", str(tmp_path / "out"),
        "--model", "vllm-local",
        "--backend", "offline",
    ])

    with _pytest.raises(SystemExit):
        runner.main()

    registry = LOCAL_MODELS["vllm-gemma-4-26b"]
    assert recorded["max_logprobs"] == runner.offline_max_logprobs(
        registry, runner.MonolithicPrompt(runner.DEFAULT_VARIANT).variant, False
    ), "the engine was left at vLLM's default of 20"
    assert recorded["timeout"] == float(registry["timeout"])
    assert recorded["model_id"] == registry["model_id"]


# ── publishing a file at all ──────────────────────────────────────────────────


def test_a_payload_ending_on_a_chunk_boundary_is_not_condemned(tmp_path):
    """A valid chunk-boundary payload must not become wreckage at 1-in-1MiB odds.

    If the check keeps only the LAST 1 MiB chunk and tests `tail.rstrip()`, a
    payload of exactly 1 (mod 1 MiB) bytes ends its final read on the lone
    trailing newline, which rstrips to empty. That classification throws away
    and regenerates 50,000 verdicts for a length.
    """
    chunk = 1 << 20
    body = b'{"a":"' + b"x" * (chunk + 1 - 9) + b'"}\n'
    assert len(body) % chunk == 1
    path = tmp_path / "verdicts-000000.json.gz"
    with gzip.open(path, "wb") as fh:
        fh.write(body)

    assert runner.published_output_is_readable(path)


def test_a_zero_length_output_is_wreckage(tmp_path, monkeypatch):
    """The FRAME CHECK catches the zero-length shapes, and each guard is pinned.

    Both shapes also fail the final-`}` test, so an assertion through only
    `published_output_is_readable` is true without either guard. Direct
    assertions independently pin `size < 20` and the all-zero-trailer test.

    `size < 20` is about a file too SHORT to have a trailer: seeking to -8 from
    the end of a 3-byte file raises OSError [Errno 22], which this runner reads
    as "the mount is broken, withhold this shard" -- a diagnosis, and the wrong
    one, about plain wreckage. 20 is the length of the smallest member this
    writer can produce (MEASURED: `GzipFile(fileobj=...)` over empty content,
    which is what `write_final_atomic` uses).

    The all-zero trailer is about COST: an empty member is 41 bytes here
    (MEASURED -- `gzip.open` on a path also writes the basename into the FNAME
    header, 20 bytes of it), so `_gzip_frame_is_plausible` must decide the
    interrupted-writer shape without decompression.
    """
    path = tmp_path / "verdicts-000000.json.gz"
    path.write_bytes(b"")
    assert not runner._gzip_frame_is_plausible(path)
    assert not runner.published_output_is_readable(path)

    truncated_header = tmp_path / "verdicts-000002.json.gz"
    truncated_header.write_bytes(b"\x1f\x8b\x08")
    assert not runner._gzip_frame_is_plausible(truncated_header), (
        "a file too short to hold a trailer was answered by seeking past its "
        "start; wreckage now reports itself as an unreadable mount"
    )

    empty_member = tmp_path / "verdicts-000001.json.gz"
    with gzip.open(empty_member, "wb"):
        pass
    assert empty_member.stat().st_size == 41, "the FNAME header moved"

    opened: list = []
    real_open = gzip.open
    monkeypatch.setattr(
        gzip, "open",
        lambda *a, **k: (opened.append(a[0]), real_open(*a, **k))[1],
    )
    assert not runner.published_output_is_readable(empty_member)
    assert not opened, (
        "an empty member was decompressed to be rejected; the all-zero trailer "
        "is what keeps the resume walk off the shard bodies"
    )


def test_a_shard_that_decompresses_short_of_the_closing_brace_is_wreckage(tmp_path):
    """What the final `}` actually decides, and nothing else could.

    This member is a COMPLETE gzip stream -- correct magic, a non-zero
    CRC32/ISIZE trailer, no error from reading it to EOF -- carrying a payload
    that stops mid-dictionary. The frame check passes it and the CRC confirms
    it, so without the closing-brace test a shard whose payload was serialized
    short reads as a finished 50,000-cell shard and the belief build joins
    against a fraction of it.
    """
    path = tmp_path / "verdicts-000000.json.gz"
    with gzip.open(path, "wb") as fh:
        fh.write(b'{"101": {"11": {"verdict": "corr')

    assert runner._gzip_frame_is_plausible(path), "premise: the frame is intact"
    with gzip.open(path, "rb") as fh:
        fh.read()  # premise: the stream verifies, so only the `}` can decide
    assert not runner.published_output_is_readable(path)


def test_a_valid_prefix_with_a_garbage_tail_is_caught_only_by_reading_it(tmp_path):
    """A valid prefix with a garbage tail requires full decompression.

    The on-disk shards were written by the pre-fsync `gzip.open` + `replace`
    writer, where a kill mid-write leaves a valid PREFIX and a tail that is not
    NULs. `_gzip_frame_is_plausible` cannot see it -- magic intact, last 8 bytes
    non-zero. Treating the full read as redundant with that symbol accepts this
    live-corpus shape.
    """
    path = tmp_path / "verdicts-000000.json.gz"
    runner.write_final_atomic(path, {"101": {"11": {"verdict": "correct"}}})
    raw = path.read_bytes()
    # body truncated, original trailer re-appended: a plausible frame over a
    # stream that ends before its end-of-stream marker
    path.write_bytes(raw[:-16] + raw[-8:])
    assert runner._gzip_frame_is_plausible(path), "premise: the frame is intact"
    assert not runner.published_output_is_readable(path)

    appended = tmp_path / "verdicts-000001.json.gz"
    appended.write_bytes(raw + b"garbage bytes!!!")
    assert runner._gzip_frame_is_plausible(appended)
    assert not runner.published_output_is_readable(appended)


def test_an_unreadable_shard_withholds_ITS_shard_and_not_the_run(tmp_path):
    """A read fault withholds only its shard; it is not file wreckage.

    `_gzip_frame_is_plausible` propagates a read fault because returning False
    makes `run_shard` regenerate ~50,000 verdicts and overwrite the published
    file with data that is NOT the same -- batched vLLM at temperature 0 is not
    bitwise reproducible on an MoE, and `_cell_priority` resolves a repeated pair
    differently from the writer that produced every file on disk.

    Raising SystemExit inside the per-shard loop aborts the run. The
    sidecar-conflict branch establishes the matching response for this per-shard
    problem: name the withheld shard and let the loop continue.
    """
    import pytest as _pytest

    jobs = _shard_jobs(3)
    outcome, final = _score_shard(tmp_path, jobs, _ScoringClient())
    assert outcome.code == 0
    before = final.read_bytes()

    final.chmod(0o000)
    try:
        with _pytest.raises(runner.ShardWithheld,
                            match="could not read the published shard"):
            runner.published_output_is_readable(final)
        rerun = _ScoringClient()
        outcome, _ = _score_shard(tmp_path, jobs, rerun)
        assert outcome.code == 2, "an unreadable file ended the whole run"
        assert "shard 7" in outcome.reason and "could not read" in outcome.reason
        assert not rerun.posts, "a shard nobody could read was rescored anyway"
    finally:
        final.chmod(0o644)
    assert final.read_bytes() == before, "the published shard was overwritten"


def test_a_sidecar_that_could_not_be_READ_is_not_a_shard_without_one(tmp_path):
    """read_shard_provenance must separate failure from genuine absence.

    Returning None after a read or parse fault makes it indistinguishable from
    "this shard predates sidecars" and routes failure onto the sidecar-less
    path, where absence never disagrees with anything. The shard is then accepted
    as matching a configuration nobody read.

    Genuine ABSENCE is the legitimate live-corpus case and must stay silent --
    the published shards have no sidecar at all.
    """
    jobs = _shard_jobs(3)
    outcome, final = _score_shard(tmp_path, jobs, _ScoringClient())
    assert outcome.code == 0
    meta = runner.meta_path_for(final)
    recorded = meta.read_bytes()

    meta.unlink()
    unrecorded = _ScoringClient()
    outcome, _ = _score_shard(tmp_path, jobs, unrecorded)
    assert outcome.code == 0 and not unrecorded.posts, (
        "the sidecar-less shards the live run published stopped being joinable"
    )

    # a sidecar whose extents were lost mid-write: a parse failure, not absence
    meta.write_bytes(recorded[: len(recorded) // 2])
    truncated = _ScoringClient()
    outcome, _ = _score_shard(tmp_path, jobs, truncated)
    assert outcome.code == 2, "an unreadable sidecar read as agreement"
    assert "shard 7" in outcome.reason and "sidecar" in outcome.reason
    assert not truncated.posts, "the shard was rescored over on a read failure"

    # the mount-fault branch
    meta.write_bytes(recorded)
    meta.chmod(0o000)
    try:
        faulted = _ScoringClient()
        outcome, _ = _score_shard(tmp_path, jobs, faulted)
        assert outcome.code == 2 and "sidecar" in outcome.reason
        assert not faulted.posts
    finally:
        meta.chmod(0o644)


def test_a_lost_tail_is_rejected_without_decompressing_the_shard(tmp_path,
                                                                 monkeypatch):
    """_gzip_frame_is_plausible rejects lost-tail wreckage without decompression.

    Decompressing every published shard is a multi-GB scan of shared scratch. The
    two shapes an interrupted writer leaves -- an empty file, and one whose tail
    extents were lost to a crash, so the CRC32/ISIZE trailer is all zeros -- are
    decidable from two seeks.
    """
    path = tmp_path / "verdicts-000000.json.gz"
    runner.write_final_atomic(path, {"1": {"2": {"verdict": "correct"}}})
    raw = bytearray(path.read_bytes())
    raw[-8:] = b"\x00" * 8
    path.write_bytes(bytes(raw))

    opened: list = []
    real_open = gzip.open

    def counting_open(*args, **kwargs):
        opened.append(args[0])
        return real_open(*args, **kwargs)

    monkeypatch.setattr(gzip, "open", counting_open)
    assert not runner.published_output_is_readable(path)
    assert not opened, (
        "the frame check did not short-circuit; every published shard would be "
        "decompressed on every resume"
    )


def test_the_sidecar_is_fsynced_before_it_is_renamed(tmp_path, monkeypatch):
    """The sidecar must be fsynced before its atomic rename.

    An unfsynced sidecar can survive a node crash as ZERO BYTES. A zero-byte
    sidecar is not a configuration and withholds its shard on every later run,
    turning a durability failure into a persistent correctness fault.
    """
    events: list[str] = []
    real_fsync = runner.os.fsync
    real_replace = Path.replace

    def record_fsync(fd):
        events.append("fsync")
        return real_fsync(fd)

    def record_replace(self, target):
        events.append("rename")
        return real_replace(self, target)

    monkeypatch.setattr(runner.os, "fsync", record_fsync)
    monkeypatch.setattr(Path, "replace", record_replace)
    runner.write_shard_meta(tmp_path / "verdicts-000000.meta.json",
                            {"model": "vllm-gemma-4-26b"})

    assert events[:2] == ["fsync", "rename"], (
        f"the sidecar was published before it was durable: {events}"
    )
    assert events.count("fsync") >= 2, "the directory entry was never fsynced"


def test_two_writers_of_one_sidecar_do_not_destroy_each_others_staging_file(
        tmp_path, monkeypatch):
    """Each sidecar writer needs a process-unique staging name.

    `path.name + ".tmp"` is derived only from the shard, so every process
    writing that sidecar can open the SAME inode. They truncate each other, and
    the loser's `replace` raises FileNotFoundError -- killing a run whose shard
    was fully scored and about to be published.
    """
    meta_path = tmp_path / "verdicts-000800.meta.json"
    both_staged = threading.Barrier(2, timeout=10)
    real_replace = Path.replace

    def gated_replace(self, target):
        both_staged.wait()
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", gated_replace)

    failures: list[BaseException] = []

    def writer(meta):
        try:
            runner.write_shard_meta(meta_path, meta)
        except BaseException as exc:  # noqa: BLE001 - the outcome under test
            failures.append(exc)

    threads = [
        threading.Thread(target=writer, args=({"variant": "verdict_only"},)),
        threading.Thread(target=writer, args=({"variant": "verdict_only"},)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(15)

    monkeypatch.undo()
    assert not failures, f"a concurrent sidecar writer died: {failures[0]!r}"
    assert json.loads(meta_path.read_text()) == {"variant": "verdict_only"}
    assert not list(tmp_path.glob("*.tmp*")), "a staging file was left behind"


# ── what the resume log is allowed to resume ──────────────────────────────────


def test_a_crash_truncated_partial_line_cannot_swallow_the_next_row(tmp_path):
    """The append log is resumed by APPENDING, and a kill lands mid-line.

    The block-buffered writer flushes ~8KB at a time, so an interruption leaves
    a final line that stops mid-JSON. Appending straight onto it concatenates
    the next scored row into that fragment: `load_partial` drops the merged line
    as undecodable, so the row is re-scored on the next resume -- and every row
    appended after it is fine, which is why this reads as ordinary rather than
    as a defect. The boundary makes the fragment its own (dropped) line.
    """
    partial = tmp_path / ".verdicts-000007.abc.partial.jsonl"
    good = {"job_id": "1:0", "verdict": "correct", "confidence": "high"}
    fragment = '{"job_id": "2:0", "verdict": "cor'
    partial.write_text(json.dumps(good) + "\n" + fragment)

    runner.ensure_append_boundary(partial)
    with partial.open("a") as fh:
        fh.write(json.dumps({"job_id": "3:0", "verdict": "incorrect",
                             "confidence": "high"}) + "\n")

    latest = runner.load_partial(partial)
    assert set(latest) == {"1:0", "3:0"}, (
        f"the row appended after a truncated line was lost: {sorted(latest)}"
    )
    assert partial.read_text().count("\n") == 3


def test_the_append_boundary_does_not_grow_a_healthy_log(tmp_path):
    """Every resume calls it, and a blank line per resume is a growing log."""
    partial = tmp_path / ".verdicts-000007.abc.partial.jsonl"
    partial.write_text('{"job_id": "1:0"}\n')
    for _ in range(3):
        runner.ensure_append_boundary(partial)
    assert partial.read_text() == '{"job_id": "1:0"}\n'

    empty = tmp_path / ".verdicts-000008.abc.partial.jsonl"
    empty.touch()
    runner.ensure_append_boundary(empty)
    assert empty.read_bytes() == b"", "a boundary was written before any row"
    runner.ensure_append_boundary(tmp_path / "never-written.partial.jsonl")


def test_a_restart_under_a_different_ceiling_cannot_resume_the_partial_log(tmp_path):
    """The resume digest must include both knobs that decide generation.

    If a shard restarts with a different --max-tokens and accepts the run-keyed
    partial log, the published shard mixes rows generated under two
    ceilings, and nothing in the file, the sidecar or the console can see it. The
    digest must also block a variant switch.
    """
    prompt = runner.MonolithicPrompt(runner.DEFAULT_VARIANT)
    # --temperature 0.7 differs from the variant's pinned 0, so provenance must
    # record the EFFECTIVE value. Comparing only with the variant attribute is
    # vacuous because _shard_args also sets 0.0; it cannot prove the digest
    # changes with the temperature that generated the rows.
    wide = runner.run_provenance(
        _shard_args(tmp_path, max_tokens=8192, temperature=0.7), prompt, None)
    narrow = runner.run_provenance(
        _shard_args(tmp_path, max_tokens=1000, temperature=0.7), prompt, None)

    assert wide["max_tokens"] == 8192 and narrow["max_tokens"] == 1000
    assert prompt.variant.temperature == 0.0
    assert wide["temperature"] == 0.0 != 0.7, (
        "the flag was recorded instead of what score_job sends, so a shard "
        "interrupted at one temperature resumes under another"
    )
    assert (runner.partial_path(tmp_path, 7, None, wide)
            != runner.partial_path(tmp_path, 7, None, narrow)), (
        "a shard interrupted at 8192 tokens resumes under a 1000-token ceiling "
        "and publishes rows from both"
    )


def test_finalize_counts_the_two_kinds_of_failure_on_the_cell_it_published(tmp_path):
    """The gate reads these counts, and a duplicate pair decides which cell wins.

    The run is keyed on job_id and the file on the (statement, evidence) pair,
    so a repeat is two rows and one cell. The class has to follow the cell that
    was actually retained -- counting the row that lost would let a transient
    duplicate withhold a shard whose published cell is a permanent failure, or
    the reverse.
    """
    shard = tmp_path / "grounded-000000.jsonl.gz"
    write_jobs(shard, [
        {"job_id": "1:0", "stmt_hash": 101, "source_hash": 11},
        {"job_id": "2:0", "stmt_hash": 202, "source_hash": 22},
        {"job_id": "3:0", "stmt_hash": 303, "source_hash": 33},
        {"job_id": "3:1", "stmt_hash": 303, "source_hash": 33},
        {"job_id": "4:0", "stmt_hash": 404, "source_hash": 44},
        {"job_id": "4:1", "stmt_hash": 404, "source_hash": 44},
    ])
    latest = {
        "1:0": {"verdict": None, "confidence": None, "error": "ReadTimeout",
                "attempts": 4, "error_class": "transient"},
        "2:0": {"verdict": None, "confidence": None, "error": "bad tier1",
                "attempts": 1, "error_class": "deterministic"},
        # two repeated pairs, in both orders: the scored read wins each time, so
        # neither pair contributes an error however the jobs happen to iterate
        "3:0": {"verdict": None, "confidence": None, "error": "ReadTimeout",
                "attempts": 4, "error_class": "transient"},
        "3:1": {"verdict": "correct", "confidence": "high"},
        "4:0": {"verdict": "correct", "confidence": "high"},
        "4:1": {"verdict": None, "confidence": None, "error": "ReadTimeout",
                "attempts": 4, "error_class": "transient"},
    }
    stats: dict = {}

    payload = runner.finalize(shard, latest, None, stats=stats)

    assert payload["303"]["33"]["verdict"] == "correct"
    assert payload["404"]["44"]["verdict"] == "correct"
    assert stats["errors_transient"] == 1, (
        "the transient failure of a pair whose OTHER row was scored is still "
        "counted toward the publication gate"
    )
    assert stats["errors_deterministic"] == 1
