"""Uniform reasoning-trace capture across backends.

Guards the model-agnostic CoT/justification separation: "did it reason"
(reasoning_tokens) is captured separately from "can we present it" (text +
status), normalized at the adapter boundary, with committed support/objection
stamped by the structured scorer. No network — HTTP is mocked.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indra_belief.model_client import (
    ModelClient, ModelResponse, ReasoningStatus,
    _build_trace, _classify_reasoning, _reasoning_tokens,
)
from indra_belief.bedrock_responses_transport import BedrockResponsesResult


# --- _classify_reasoning -----------------------------------------------------

def test_classify_plaintext():
    assert _classify_reasoning("some cot", 100, inline=False) == ReasoningStatus.PLAINTEXT

def test_classify_inline():
    assert _classify_reasoning("cot in content", -1, inline=True) == ReasoningStatus.INLINE

def test_classify_not_returned():
    # reasoned (tokens>0) but no readable text → suppressed, not "none"
    assert _classify_reasoning("", 1847, inline=False) == ReasoningStatus.NOT_RETURNED

def test_classify_none():
    assert _classify_reasoning("", 0, inline=False) == ReasoningStatus.NONE
    assert _classify_reasoning("", -1, inline=False) == ReasoningStatus.NONE


# --- _reasoning_tokens: parity across the two provider shapes -----------------

class _Details:
    def __init__(self, n): self.reasoning_tokens = n

class _Usage:
    def __init__(self, n): self.completion_tokens_details = _Details(n)

def test_reasoning_tokens_openai_object():
    assert _reasoning_tokens(_Usage(123)) == 123

def test_reasoning_tokens_bedrock_dict():
    assert _reasoning_tokens({"output_tokens_details": {"reasoning_tokens": 123}}) == 123

def test_reasoning_tokens_missing_returns_minus1():
    assert _reasoning_tokens({}) == -1
    assert _reasoning_tokens(None) == -1
    class _Bare: pass
    assert _reasoning_tokens(_Bare()) == -1


# --- ModelResponse backward-compat + JSON-serializability --------------------

def test_modelresponse_default_trace_is_none_status():
    r = ModelResponse(content="x", reasoning="", tokens=1, raw_text="x", finish_reason="stop")
    assert r.reasoning_trace["status"] == ReasoningStatus.NONE
    assert r.reasoning_trace["committed_justification"] == {
        "support": None, "objection": None, "source": None}
    json.dumps(r.reasoning_trace)  # must serialize (it lands in the call log / record)

def test_build_trace_shape():
    t = _build_trace(reasoning="cot", reasoning_tokens=5, status=ReasoningStatus.PLAINTEXT,
                     provider_source="x", backend="b", model_id="m", finish_reason="stop")
    assert set(t) == {"free_cot", "status", "reasoning_tokens", "provider_source",
                      "observed_message_keys", "backend", "model_id", "finish_reason",
                      "committed_justification"}
    assert t["free_cot"] == "cot" and t["status"] == "plaintext"
    # Reasoning was captured, so there is nothing to diagnose and the key names
    # are not recorded. They are only carried on the absence path.
    assert t["observed_message_keys"] == []


def test_build_trace_records_what_the_reply_carried_when_nothing_reasoned():
    """`status: "none"` has two causes and used to be written for both.

    Either the model did not reason, or the reply put its reasoning under a field
    this code did not read. Measured on the shipped GLM-5 arm, the second was the
    true one on every sampled call — 2,663/2,663 in `glm_5_primary` recorded an
    empty channel while the median call billed 433 output tokens against a
    272-character answer. Recording the reply's own key names makes the two
    distinguishable from the durable record.
    """
    t = _build_trace(reasoning="", reasoning_tokens=-1, status=ReasoningStatus.NONE,
                     provider_source="", backend="b", model_id="m", finish_reason="stop",
                     observed_message_keys=("content", "reasoning", "role"))
    assert t["provider_source"] == ""
    assert t["observed_message_keys"] == ["content", "reasoning", "role"]
    json.dumps(t)  # it lands in the call log, so it must serialize


# --- bedrock_responses adapter: encrypted vs plaintext (mock HTTP) -----------

def _fake_urlopen(payload):
    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps(payload).encode()
    def _open(req, timeout=None): return _Resp()
    return _open

def _install_fake_bedrock_open(c, payload):
    class _Opener:
        open = staticmethod(_fake_urlopen(payload))
    c._bedrock_url_opener = _Opener()

def _responses_client(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-token")
    return ModelClient("bedrock-gpt-5.5")  # backend == bedrock_responses

def test_responses_encrypted(monkeypatch):
    c = _responses_client(monkeypatch)
    payload = {"status": "completed",
               "usage": {"output_tokens": 40, "input_tokens": 10,
                         "output_tokens_details": {"reasoning_tokens": 1847}},
               "output": [{"type": "reasoning", "summary": [], "id": "r"},
                          {"type": "message",
                           "content": [{"type": "output_text", "text": '{"verdict":"correct"}'}]}]}
    _install_fake_bedrock_open(c, payload)
    resp = c._call_bedrock_responses("sys", [{"role": "user", "content": "hi"}], 100, 0.1, 30,
                                     reasoning_effort="high")
    assert resp.reasoning_trace["status"] == ReasoningStatus.ENCRYPTED
    assert resp.reasoning_trace["reasoning_tokens"] == 1847
    assert resp.reasoning == ""
    assert resp.content == '{"verdict":"correct"}'   # answer still extracted

def test_responses_plaintext(monkeypatch):
    c = _responses_client(monkeypatch)
    payload = {"status": "completed", "usage": {"output_tokens": 40, "input_tokens": 10},
               "output": [{"type": "reasoning", "content": [{"text": "step by step CoT"}], "id": "r"},
                          {"type": "message",
                           "content": [{"type": "output_text", "text": '{"verdict":"correct"}'}]}]}
    _install_fake_bedrock_open(c, payload)
    resp = c._call_bedrock_responses("sys", [{"role": "user", "content": "hi"}], 100, 0.1, 30,
                                     reasoning_effort="high")
    assert resp.reasoning_trace["status"] == ReasoningStatus.PLAINTEXT
    assert resp.reasoning == "step by step CoT"


def _raw_responses_client(result: BedrockResponsesResult):
    class _Transport:
        def call(self, _body, *, timeout):
            assert timeout == 30
            return result

    client = ModelClient.__new__(ModelClient)
    client.model_name = "bedrock-gemma-4-e2b-fixture"
    client.backend = "bedrock_responses_raw"
    client.config = {
        "model_id": "google.gemma-4-e2b",
        "reasoning_in_content": False,
        "reasoning_effort": "high",
    }
    client._bedrock_responses_transport = _Transport()
    return client


def test_raw_responses_plaintext_reasoning_classification():
    result = BedrockResponsesResult(
        content='{"verdict":"correct"}',
        reasoning="step by step CoT",
        prompt_tokens=10,
        output_tokens=40,
        reasoning_tokens=17,
        finish_reason="stop",
        reasoning_item_present=True,
        transport_trace={"request_body_sha256": "a" * 64},
    )
    response = _raw_responses_client(result)._call_bedrock_responses_raw(
        "sys", [{"role": "user", "content": "hi"}], 100, 0.1, 30,
        reasoning_effort="high",
    )
    assert response.reasoning_trace["status"] == ReasoningStatus.PLAINTEXT
    assert response.reasoning_trace["backend"] == "bedrock_responses_raw"
    assert response.raw_text == "step by step CoT\n" + response.content
    assert response.transport_trace == result.transport_trace


def test_raw_responses_encrypted_reasoning_classification():
    result = BedrockResponsesResult(
        content='{"verdict":"correct"}',
        reasoning="",
        prompt_tokens=10,
        output_tokens=40,
        reasoning_tokens=1847,
        finish_reason="stop",
        reasoning_item_present=True,
        transport_trace={"request_body_sha256": "b" * 64},
    )
    response = _raw_responses_client(result)._call_bedrock_responses_raw(
        "sys", [{"role": "user", "content": "hi"}], 100, 0.1, 30,
        reasoning_effort="high",
    )
    assert response.reasoning_trace["status"] == ReasoningStatus.ENCRYPTED
    assert response.reasoning_trace["reasoning_tokens"] == 1847
    assert response.raw_text == response.content


# --- scorer stamps committed support/objection onto the trace ----------------

def test_compact_reasoning_trace_export_projection():
    from indra_belief.results import compact_reasoning_trace, _FREE_COT_CLIP
    assert compact_reasoning_trace(None) is None          # legacy row → None
    assert compact_reasoning_trace("nope") is None
    raw = _build_trace(reasoning="x" * (_FREE_COT_CLIP + 500), reasoning_tokens=172,
                       status=ReasoningStatus.ENCRYPTED, provider_source="p",
                       backend="bedrock_responses", model_id="m", finish_reason="stop")
    raw["committed_justification"] = {"support": "S", "objection": "O", "source": "answer_json"}
    out = compact_reasoning_trace(raw)
    assert out["status"] == "encrypted" and out["reasoning_tokens"] == 172
    assert len(out["free_cot"]) == _FREE_COT_CLIP          # clipped
    assert out["free_cot_chars"] == _FREE_COT_CLIP + 500   # full length preserved
    assert out["committed_justification"] == {"support": "S", "objection": "O", "source": "answer_json"}


def test_scorer_stamps_committed_justification():
    from indra_belief.scorers.monolithic import scorer as S
    # Pass the variant explicitly: the stamping branch is a property of the
    # variant, not of the ambient MONO_VARIANT this process happens to carry.
    variant = S.VARIANTS["disconfirm_relnature_rf"]
    assert variant.structured
    js = ('{"support":"X binds Y","objection":"amount not activity",'
          '"verdict":"incorrect","confidence":"high"}')
    r = ModelResponse(content=js, reasoning="", tokens=1, raw_text=js, finish_reason="stop")
    S._stamp_committed_justification(r, variant=variant)
    cj = r.reasoning_trace["committed_justification"]
    assert cj["support"] == "X binds Y"
    assert cj["objection"] == "amount not activity"
    assert cj["source"] == "answer_json"
