"""The corpus-scale runner sends what the variant declares — including no CoT.

WHY THIS EXISTS
---------------
The shard runner drives a bare ``httpx.Client``, not a ``ModelClient``, so none
of the transport guarantees the live path enjoys apply to it automatically. Its
main-call body was, verbatim::

    {"model", "messages", "max_tokens", "temperature"}

There is no reasoning control in that dict. The thinking channel was therefore
whatever the served chat template happened to default to — ON, for gemma-4 — on
the one path built to read 60M evidences. Nothing failed; the runs just
deliberated, and the only symptom was the bill.

The fix is not a no-thinking flag. Suppressing reasoning while still sending the
DELIBERATIVE prompt is a MEASURED silent failure: the verdict lands 56 tokens
deep at delta_logit +22.50, indistinguishable from full deliberation
(probes/reader.py). Prompt shape, reasoning channel and sampling temperature are
a coherent set, and a variant is the object that carries all three. So these
tests assert the SET, not any one key.

Everything here runs against a local stub server that records the bodies it
receives. That is deliberate: the defect was in what went ON THE WIRE, and a
test that inspects Python objects instead of the request would have passed
throughout the period the runner was silently paying for CoT.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx")

ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "shard_runner_under_test", ROOT / "scripts/run_vllm_processed_shards.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


# A reply shaped like a real one: the verdict first, and logprobs at that
# position carrying both labels, so the free in-call margin is readable.
def _reply(content: str, *, with_logprobs: bool) -> dict:
    message = {"role": "assistant", "content": content}
    choice: dict = {"message": message, "finish_reason": "stop"}
    if with_logprobs:
        choice["logprobs"] = {
            "content": [
                {"token": "correct", "logprob": -0.05,
                 "top_logprobs": [{"token": "correct", "logprob": -0.05},
                                  {"token": "incorrect", "logprob": -3.05}]},
            ]
        }
    return {"choices": [choice], "usage": {"completion_tokens": 5}}


class _Recorder(BaseHTTPRequestHandler):
    bodies: list = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).bodies.append(body)
        wants_logprobs = bool(body.get("logprobs"))
        payload = _reply('{"verdict": "correct"}', with_logprobs=wants_logprobs)
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):  # silence
        pass


@pytest.fixture
def stub():
    """A recording server. Yields (endpoint, bodies)."""

    class Handler(_Recorder):
        bodies: list = []

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/chat/completions", Handler.bodies
    finally:
        server.shutdown()
        server.server_close()


JOB = {
    "job_id": "t:0",
    "stmt_hash": "1",
    "source_hash": "a",
    "stmt_type": "Activation",
    "needs_llm": True,
    "subject": "IL18",
    "object": "RAF1",
    "evidence_text": "IL-18 induced activation of Raf-1 in NK cells.",
    "user_message": "Statement: Activation(IL18, RAF1)\nEvidence: IL-18 ...",
}


def _score(endpoint, variant, **kw):
    prompt = runner.MonolithicPrompt(variant)
    with httpx.Client(timeout=30) as client:
        return runner.score_job(
            dict(JOB), client=client, prompt=prompt, endpoint=endpoint,
            model_id="stub-model", max_tokens=256,
            temperature=kw.pop("temperature", 0.1), **kw,
        )


# ── the defect itself ─────────────────────────────────────────────────────────

def test_the_default_run_suppresses_reasoning_on_the_wire(stub):
    """The regression. Both mechanisms must be present, because either alone is
    a silent no-op on some substrate: vLLM/Ollama-served Gemma DROPS
    `reasoning_effort="none"` and honors `chat_template_kwargs`, while other
    backends do the reverse."""
    endpoint, bodies = stub
    _score(endpoint, runner.DEFAULT_VARIANT)
    assert bodies, "the runner issued no request at all"
    body = bodies[0]
    assert body.get("reasoning_effort") == "none", body.keys()
    assert body.get("chat_template_kwargs") == {"enable_thinking": False}, (
        "the chat_template_kwargs mechanism is missing; on a vLLM-served Gemma "
        "this run deliberates at full cost and nothing reports that it did"
    )


def test_the_default_variant_is_a_no_cot_one(stub):
    """A default of 'whatever the template does' is what produced the defect."""
    prompt = runner.MonolithicPrompt(runner.DEFAULT_VARIANT)
    assert prompt.variant.reasoning_effort == "none"


def test_temperature_comes_from_the_variant_not_the_cli_default(stub):
    """The in-call label read is only valid at temperature 0, so the variant
    overrides the 0.1 the CLI defaults to."""
    endpoint, bodies = stub
    _score(endpoint, "verdict_only", temperature=0.1)
    assert bodies[0]["temperature"] == 0.0


def test_a_cli_temperature_cannot_silently_invalidate_the_margin(stub):
    """The invariant ModelClient enforces, enforced here too. Above 0 the
    reported argmax stream can diverge from the sampled text, so the verdict
    POSITION is untrustworthy and the margin becomes a plausible-looking lie."""
    endpoint, _ = stub
    prompt = runner.MonolithicPrompt("verdict_only")
    prompt.variant = prompt.variant.__class__(
        **{**prompt.variant.__dict__, "temperature": None}
    )
    with httpx.Client(timeout=30) as client:
        with pytest.raises(SystemExit, match="only valid at 0"):
            runner.score_job(
                dict(JOB), client=client, prompt=prompt, endpoint=endpoint,
                model_id="stub", max_tokens=256, temperature=0.7,
            )


# ── the free margin ───────────────────────────────────────────────────────────

def test_the_margin_is_read_from_the_scoring_call_itself(stub):
    endpoint, bodies = stub
    row = _score(endpoint, "verdict_only")
    assert row["verdict"] == "correct"
    assert row["probe_delta_logit"] == pytest.approx(3.0, abs=1e-6), (
        "expected log P(correct) - log P(incorrect) = -0.05 - -3.05"
    )
    assert bodies[0].get("logprobs") is True
    assert bodies[0].get("top_logprobs") == 128


def test_the_free_margin_costs_exactly_one_request(stub):
    """The whole point at 60M evidences. `--probe`'s second call would double
    the request count for a strictly worse reading (n=80: in-call AUROC 0.8734,
    probe 0.7237), so it must not fire when the margin is already in hand."""
    endpoint, bodies = stub
    _score(endpoint, "verdict_only", probe=True)
    assert len(bodies) == 1, (
        f"{len(bodies)} requests issued; the in-call margin was available and a "
        "second probe call was made anyway"
    )


def test_a_failed_margin_read_never_costs_the_verdict(stub):
    """The margin is an extra measurement riding along, not a precondition."""
    endpoint, _ = stub
    from indra_belief.probes import reader

    assert reader.label_margin_from_payload({"choices": []}) is None
    row = _score(endpoint, "verdict_only")
    assert row["verdict"] == "correct"


# ── the deliberative path is untouched ────────────────────────────────────────

def test_the_reasoning_variant_still_reasons(stub):
    """The no-CoT default must not silently disable deliberation for a caller
    who explicitly selected the reasoning-first variant for a gold-eval run."""
    endpoint, bodies = stub
    _score(endpoint, "disconfirm_relnature_rf", temperature=0.1)
    main = bodies[-1]
    assert "chat_template_kwargs" not in main, (
        "the deliberative variant had its thinking channel suppressed"
    )
    assert main["temperature"] == 0.1, "the CLI temperature should still apply"
    assert "top_logprobs" not in main, (
        "a logprob window must not leak onto a call whose verdict is not first — "
        "it would read the label ~56 tokens deep and return a saturated number"
    )


def test_a_variant_with_no_relation_resolver_skips_the_relation_call(stub):
    """verdict_only has no prompt to consume a relation note, so paying for the
    extra Complex call would buy nothing."""
    endpoint, bodies = stub
    complex_job = {**JOB, "stmt_type": "Complex"}
    prompt = runner.MonolithicPrompt("verdict_only")
    with httpx.Client(timeout=30) as client:
        runner.score_job(complex_job, client=client, prompt=prompt,
                         endpoint=endpoint, model_id="stub", max_tokens=256,
                         temperature=0.1)
    assert len(bodies) == 1, "verdict_only issued a relation-nature call"


# ── the banner must describe the run that happens ─────────────────────────────

def test_the_calibration_banner_hashes_the_prompt_actually_sent():
    """While the runner pinned one prompt this was trivially true. Once
    --variant could change it, a re-imported constant would report the
    calibration status of a prompt the run never used."""
    import hashlib

    from indra_belief.calibration_constants import calibration_banner

    seen = set()
    for variant in ("verdict_only", "disconfirm_relnature_rf"):
        prompt = runner.MonolithicPrompt(variant)
        sha = hashlib.sha256(prompt.system_prompt.encode()).hexdigest()
        seen.add(sha)
        _, banner = calibration_banner("vllm-local", sha)
        assert sha[:12] in banner, "the banner names a different prompt"
    assert len(seen) == 2, "two variants hashed identically; the pin is still in"


def test_an_unknown_variant_fails_before_any_shard_is_opened():
    with pytest.raises(SystemExit, match="unknown --variant"):
        runner.MonolithicPrompt("no-such-variant")


# ── the preflight ─────────────────────────────────────────────────────────────
#
# The no-CoT variant puts `logprobs: true` + a 128-wide window on EVERY scoring
# request, because the verdict and its margin come from the same call. vLLM caps
# that at --max-logprobs, whose default is far below 128. A server started
# without the flag therefore rejects the whole run, not just the margin — so the
# runner proves the server agrees before it opens a single shard.

class _Rejecter(BaseHTTPRequestHandler):
    """A server that refuses a logprob window, the way a default vLLM does."""

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if body.get("top_logprobs"):
            data = json.dumps({"error": {
                "message": "top_logprobs must be <= 20, the value of --max-logprobs"
            }}).encode()
            self.send_response(400)
        else:
            data = json.dumps(_reply("ok", with_logprobs=False)).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


class _Silent(BaseHTTPRequestHandler):
    """Accepts the request and returns no logprobs at all — the quiet failure."""

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        data = json.dumps(_reply("ok", with_logprobs=False)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def _serve(handler):
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/v1/chat/completions"


def _preflight(handler, variant=None):
    server, endpoint = _serve(handler)
    prompt = runner.MonolithicPrompt(variant or runner.DEFAULT_VARIANT)
    try:
        with httpx.Client(timeout=30) as client:
            runner.preflight(client, endpoint, "stub-model", prompt)
    finally:
        server.shutdown()
        server.server_close()


def test_preflight_names_the_exact_server_flag_when_the_window_is_refused():
    """The message has to carry the remedy. 'HTTP 400' alone sends someone
    reading vLLM docs; the number 128 is ours, not theirs."""
    with pytest.raises(SystemExit) as exc:
        _preflight(_Rejecter)
    message = str(exc.value)
    assert "--max-logprobs 128" in message, message
    assert "400" in message


def test_preflight_rejects_a_server_that_silently_drops_logprobs():
    """The worse failure: 200 OK, verdicts fine, every margin null, and a run
    that looks completely healthy while collecting nothing."""
    with pytest.raises(SystemExit, match="NO logprobs"):
        _preflight(_Silent)


def test_preflight_passes_on_a_server_that_agrees(capsys):
    _preflight(_Recorder)
    assert "[preflight] ok" in capsys.readouterr().out


def test_preflight_is_satisfied_by_a_plain_server_for_a_no_logprob_variant():
    """The deliberative variant asks for no window, so a stock server is fine —
    the check must not invent a requirement the run does not have."""
    _preflight(_Silent, variant="disconfirm_relnature_rf")


def test_preflight_reports_an_unreachable_server_as_such():
    prompt = runner.MonolithicPrompt(runner.DEFAULT_VARIANT)
    with httpx.Client(timeout=5) as client:
        with pytest.raises(SystemExit, match="cannot reach"):
            runner.preflight(client, "http://127.0.0.1:9/v1/chat/completions",
                             "stub", prompt)
