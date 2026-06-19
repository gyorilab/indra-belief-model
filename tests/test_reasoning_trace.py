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
                      "backend", "model_id", "finish_reason", "committed_justification"}
    assert t["free_cot"] == "cot" and t["status"] == "plaintext"


# --- bedrock_responses adapter: encrypted vs plaintext (mock HTTP) -----------

def _fake_urlopen(payload):
    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps(payload).encode()
    def _open(req, timeout=None): return _Resp()
    return _open

def _responses_client(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-token")
    return ModelClient("bedrock-gpt-5.5")  # backend == bedrock_responses

def test_responses_encrypted(monkeypatch):
    import urllib.request
    c = _responses_client(monkeypatch)
    payload = {"status": "completed",
               "usage": {"output_tokens": 40, "input_tokens": 10,
                         "output_tokens_details": {"reasoning_tokens": 1847}},
               "output": [{"type": "reasoning", "summary": [], "id": "r"},
                          {"type": "message",
                           "content": [{"type": "output_text", "text": '{"verdict":"correct"}'}]}]}
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen(payload))
    resp = c._call_bedrock_responses("sys", [{"role": "user", "content": "hi"}], 100, 0.1, 30,
                                     reasoning_effort="high")
    assert resp.reasoning_trace["status"] == ReasoningStatus.ENCRYPTED
    assert resp.reasoning_trace["reasoning_tokens"] == 1847
    assert resp.reasoning == ""
    assert resp.content == '{"verdict":"correct"}'   # answer still extracted

def test_responses_plaintext(monkeypatch):
    import urllib.request
    c = _responses_client(monkeypatch)
    payload = {"status": "completed", "usage": {"output_tokens": 40, "input_tokens": 10},
               "output": [{"type": "reasoning", "content": [{"text": "step by step CoT"}], "id": "r"},
                          {"type": "message",
                           "content": [{"type": "output_text", "text": '{"verdict":"correct"}'}]}]}
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen(payload))
    resp = c._call_bedrock_responses("sys", [{"role": "user", "content": "hi"}], 100, 0.1, 30,
                                     reasoning_effort="high")
    assert resp.reasoning_trace["status"] == ReasoningStatus.PLAINTEXT
    assert resp.reasoning == "step by step CoT"


# --- scorer stamps committed support/objection onto the trace ----------------

def test_scorer_stamps_committed_justification():
    from indra_belief.scorers.monolithic import scorer as S
    if S._VARIANT not in S._STRUCTURED_VARIANTS:
        pytest.skip("committed_justification only applies to structured variants")
    js = ('{"support":"X binds Y","objection":"amount not activity",'
          '"verdict":"incorrect","confidence":"high"}')
    r = ModelResponse(content=js, reasoning="", tokens=1, raw_text=js, finish_reason="stop")
    S._stamp_committed_justification(r)
    cj = r.reasoning_trace["committed_justification"]
    assert cj["support"] == "X binds Y"
    assert cj["objection"] == "amount not activity"
    assert cj["source"] == "answer_json"
