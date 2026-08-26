"""ModelClient coverage for the in-process vLLM transport."""
from __future__ import annotations

import importlib.util
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.model_client import ModelClient  # noqa: E402


# THIS STUB PINS OUR DISPATCHER, NOT THE ENGINE. It defines the vLLM shapes it
# later asserts against, so it proves lazy construction, request translation,
# batching, and response translation only. It does not exercise real vLLM.
class _SamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _Logprob:
    def __init__(self, token: str, value: float):
        self.decoded_token = token
        self.logprob = value


class _Completion:
    text = "correct"
    token_ids = [1]
    logprobs = [
        {
            1: _Logprob("correct", -0.1),
            2: _Logprob("incorrect", -2.1),
        }
    ]
    finish_reason = "stop"


class _Output:
    outputs = [_Completion()]
    prompt_token_ids = [10, 11]


def _install_vllm(monkeypatch, *, hold_first_chat: bool = False,
                  structured: object | None = None) -> dict:
    state = {
        "instances": [],
        "calls": [],
        "first_chat_entered": threading.Event(),
        "release_first_chat": threading.Event(),
    }

    class LLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            state["instances"].append(self)

        def chat(self, conversations, sampling_params, use_tqdm, **kwargs):
            call = {
                "conversations": conversations,
                "sampling_params": sampling_params,
                "use_tqdm": use_tqdm,
                "kwargs": kwargs,
            }
            state["calls"].append(call)
            if hold_first_chat and len(state["calls"]) == 1:
                state["first_chat_entered"].set()
                assert state["release_first_chat"].wait(5)
            return [_Output() for _ in conversations]

    vllm = types.SimpleNamespace(LLM=LLM, SamplingParams=_SamplingParams)
    sampling_params = structured or types.SimpleNamespace()
    vllm.sampling_params = sampling_params
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_params)
    return state


def _call(client: ModelClient, **kwargs):
    return client.call(
        "system",
        [{"role": "user", "content": "question"}],
        **kwargs,
    )


def _close(client: ModelClient) -> None:
    if client._client is not None:
        client._client.close()


def test_constructor_is_lazy_and_first_call_uses_the_registry_engine_window(
    monkeypatch,
):
    state = _install_vllm(monkeypatch)

    client = ModelClient("vllm-offline-gemma-4-26b")
    assert client.backend == "vllm_offline"
    assert state["instances"] == []
    assert client._client is None

    try:
        response = _call(
            client,
            temperature=0,
            top_logprobs=2048,
            reasoning_effort="none",
        )
    finally:
        _close(client)

    assert len(state["instances"]) == 1
    assert state["instances"][0].kwargs["max_logprobs"] == 1024
    assert state["instances"][0].kwargs["gpu_memory_utilization"] == 0.9
    params = state["calls"][0]["sampling_params"][0].kwargs
    assert params["logprobs"] == 1024
    assert state["calls"][0]["kwargs"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert response.logprobs_status == "ok"
    assert response.logprobs[0]["token"] == "correct"
    assert {row["token"] for row in response.logprobs[0]["top"]} == {
        "correct",
        "incorrect",
    }


def test_two_concurrent_model_calls_land_in_one_llm_chat(monkeypatch):
    state = _install_vllm(monkeypatch, hold_first_chat=True)
    client = ModelClient("vllm-offline-gemma-4-26b")

    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            warm = pool.submit(_call, client, temperature=0)
            assert state["first_chat_entered"].wait(5)
            first = pool.submit(_call, client, temperature=0)
            second = pool.submit(_call, client, temperature=0)

            deadline = time.monotonic() + 5
            while client._client.requests.qsize() < 2:
                assert time.monotonic() < deadline, "concurrent calls never queued"
                time.sleep(0.001)
            state["release_first_chat"].set()

            warm.result(timeout=5)
            first.result(timeout=5)
            second.result(timeout=5)
    finally:
        state["release_first_chat"].set()
        _close(client)

    batched_calls = state["calls"][1:]
    assert len(batched_calls) == 1
    assert len(batched_calls[0]["conversations"]) == 2


def test_top_logprobs_above_temperature_zero_still_raises(monkeypatch):
    state = _install_vllm(monkeypatch)
    client = ModelClient("vllm-offline-gemma-4-26b")

    with pytest.raises(ValueError, match="top_logprobs requires temperature=0"):
        _call(client, temperature=0.1, top_logprobs=5)

    assert state["instances"] == []


def test_json_object_response_format_reaches_sampling_params(monkeypatch):
    class GuidedDecodingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    state = _install_vllm(
        monkeypatch,
        structured=types.SimpleNamespace(
            GuidedDecodingParams=GuidedDecodingParams
        ),
    )
    client = ModelClient("vllm-offline-gemma-4-26b")

    try:
        _call(
            client,
            response_format={"type": "json_object"},
        )
    finally:
        _close(client)

    params = state["calls"][0]["sampling_params"][0].kwargs
    assert params["guided_decoding"].kwargs == {"json_object": True}


def test_shard_runner_rejects_the_modelclient_only_registry_entry(monkeypatch):
    script = ROOT / "scripts" / "run_vllm_processed_shards.py"
    spec = importlib.util.spec_from_file_location("_offline_key_guard", script)
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    def shard_discovery_must_not_run(*_args, **_kwargs):
        raise AssertionError("the mismatch guard ran after shard discovery")

    monkeypatch.setattr(runner, "select_shards", shard_discovery_must_not_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_vllm_processed_shards.py",
            "--model",
            "vllm-offline-gemma-4-26b",
        ],
    )

    with pytest.raises(SystemExit) as raised:
        runner.main()

    message = str(raised.value)
    assert "vllm-offline-gemma-4-26b" in message
    assert "vllm_offline" in message
    assert "vllm-gemma-4-26b" in message
