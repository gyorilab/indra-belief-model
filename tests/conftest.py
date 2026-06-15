"""Shared pytest fixtures and test helpers.

``_MockResponse`` is the minimal ModelResponse-shaped object used by the
mock clients in test_probes.py and test_orchestrator.py. It lives here so
both modules import one definition instead of carrying byte-identical
copies. The two ``_MockClient`` classes stay separate — they track
different things — but the response shape they return is identical.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _MockResponse:
    content: str
    raw_text: str = ""
    tokens: int = 10
    finish_reason: str = "stop"
    reasoning: str = ""
    prompt_tokens: int = 100
