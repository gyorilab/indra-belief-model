"""In-process offline vLLM transport for raw OpenAI-compatible requests.

Option (b): keep batching inside this TRANSPORT, which duck-types the httpx
surface the shard runner uses (``post`` / ``raise_for_status`` / ``json``). Its
queue and 2 ms worker gather window turn concurrent one-at-a-time posts into
one ``llm.chat`` call per template group; that invisible batching is why the
transport exists.

Option (a), adding a ``call_batch`` seam to ``ModelClient``, buys nothing. The
batching seam already exists inside this transport, while a batch API would
force the shard runner's per-job pipeline -- per-row retry, ``.partial.jsonl``
append, and per-row failure isolation -- to be rewritten on the 60M path for no
behavioral gain. The UNEXERCISED AGAINST REAL vLLM caveat below applies to every
caller of this transport, including ``ModelClient``.
"""
from __future__ import annotations

import concurrent.futures as cf
import queue
import threading
import time
from typing import Any


class _OfflineResponse:
    """Small response adapter matching the httpx methods used below."""

    status_code = 200
    text = ""

    def __init__(self, payload: dict[str, Any]):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


def _settle_exception(future: cf.Future, exc: BaseException) -> None:
    """Fail a future unless it was already resolved.

    `set_exception` on a FINISHED future raises InvalidStateError, and that
    batch failure handler walks every future, including ones it already
    resolved. An unguarded call therefore raises inside the handler, kills the
    dispatcher thread, and leaves the shard hung with its scored rows still in
    RAM while later `post` calls enqueue into a queue nobody drains.
    """
    if not future.done():
        future.set_exception(exc)


def _template_signature(payload: dict[str, Any]) -> tuple:
    """The chat arguments `llm.chat()` applies to a WHOLE call, not per row."""
    template = payload.get("chat_template_kwargs")
    return (
        tuple(sorted(template.items())) if isinstance(template, dict) else None,
        payload.get("continue_final_message"),
        payload.get("add_generation_prompt"),
    )


def _chat_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate one payload's whole-call arguments for ``llm.chat``.

    `continue_final_message` and `add_generation_prompt` are as load-bearing as
    the template kwargs. `build_probe_request` sets them to keep the assistant
    PREFILL open, which is the only reason the verdict sits at generated
    position 0. Without them, the model opens a fresh turn and the probe reads a
    position that is not the label.
    """
    kwargs: dict[str, Any] = {}
    template = payload.get("chat_template_kwargs")
    if isinstance(template, dict):
        kwargs["chat_template_kwargs"] = template
    for key in ("continue_final_message", "add_generation_prompt"):
        if key in payload:
            kwargs[key] = bool(payload[key])
    return kwargs


def _structured_output_support() -> tuple[str, Any] | None:
    """How THIS vLLM spells `response_format: {"type": "json_object"}`.

    vLLM exposes both spellings (`guided_decoding` and `structured_outputs`)
    across supported installs. The caller raises if neither exists rather than
    degrading to an unconstrained reply.
    """
    try:
        from vllm import sampling_params as vllm_sampling_params
    except Exception:
        return None
    for attribute, keyword in (
        ("StructuredOutputsParams", "structured_outputs"),
        ("GuidedDecodingParams", "guided_decoding"),
    ):
        params_cls = getattr(vllm_sampling_params, attribute, None)
        if params_cls is not None:
            return keyword, params_cls
    return None


class OfflineVllmClient:
    """UNEXERCISED AGAINST REAL vLLM: batch posts through one long-lived vllm.LLM.

    No machine in this repo's reach can load a 26B model, so every test of this
    class installs its own `sys.modules['vllm']` stub and therefore asserts
    against shapes the test file itself defines -- the batching, the failure
    isolation and the response translation are pinned, the engine's actual
    behaviour is not. The corpus path runs --backend server, where each job is
    its own request; treat this backend as unvalidated until someone runs it on
    a GPU.
    """

    # Preflight reports the transport it could not reach. An offline run
    # contacts no HTTP endpoint, and sending the operator to check --base-url
    # after a 26B model has already been loaded diagnoses the wrong thing.
    transport_description = "the in-process vLLM engine (--backend offline)"

    def __init__(
        self,
        model_id: str,
        *,
        batch_size: int,
        gpu_memory_utilization: float,
        max_logprobs: int | None = None,
        timeout: float | None = None,
    ):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise SystemExit(
                "offline backend requires vLLM: python -m pip install vllm"
            ) from exc

        self.sampling_params_cls = SamplingParams
        self.structured_outputs = _structured_output_support()
        self.batch_size = batch_size
        self.timeout = timeout
        self.requests: queue.Queue = queue.Queue()
        self.failure: BaseException | None = None
        engine_kwargs: dict[str, Any] = {}
        if max_logprobs:
            # The offline mirror of the server's `--max-logprobs`. vLLM's own
            # default is 20 and the shipped variant asks for a 128-wide window,
            # so without this the engine rejects EVERY request -- after the
            # model is loaded, and by way of a preflight message about an HTTP
            # server this backend never contacts.
            engine_kwargs["max_logprobs"] = int(max_logprobs)
        self.llm = LLM(
            model=model_id,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            **engine_kwargs,
        )
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def post(
        self,
        _endpoint: str,
        *,
        json: dict[str, Any],
        timeout: float | None = None,
    ) -> _OfflineResponse:
        if not self.worker.is_alive():
            raise RuntimeError(
                "the offline vLLM dispatcher is not running"
                + (f": {type(self.failure).__name__}: {self.failure}"
                   if self.failure is not None else "")
            )
        future: cf.Future = cf.Future()
        self.requests.put((json, future))
        # BOUNDED. An unbounded future.result() turns any dispatcher failure into
        # a hung run: every worker parks here, the pool never drains, and the
        # shard produces no output at all rather than a per-job error.
        return future.result(timeout=self.timeout if timeout is None else timeout)

    @staticmethod
    def _openai_logprobs(completion) -> dict[str, Any] | None:
        """Translate vLLM offline token logprobs to the OpenAI response shape."""
        steps = getattr(completion, "logprobs", None)
        if not steps:
            return None
        content: list[dict[str, Any]] = []
        for token_id, alternatives in zip(completion.token_ids, steps):
            emitted = alternatives.get(token_id)
            if emitted is None:
                continue
            content.append(
                {
                    "token": str(getattr(emitted, "decoded_token", "") or ""),
                    "logprob": float(emitted.logprob),
                    "top_logprobs": [
                        {
                            "token": str(
                                getattr(candidate, "decoded_token", "") or ""
                            ),
                            "logprob": float(candidate.logprob),
                        }
                        for candidate in alternatives.values()
                    ],
                }
            )
        return {"content": content}

    def _sampling_params(self, payload: dict[str, Any]):
        kwargs: dict[str, Any] = {
            "temperature": float(payload.get("temperature", 0.1)),
            "max_tokens": int(payload.get("max_tokens", 1000)),
            "logprobs": (
                int(payload["top_logprobs"]) if payload.get("logprobs") else None
            ),
        }
        response_format = payload.get("response_format") or {}
        if response_format.get("type") == "json_object":
            # The relation-nature call forces a JSON reply. Dropping that here
            # yields free text: the object does not parse, `relation_note`
            # returns "", and the [Complex] claim is scored without the rejection
            # note the variant exists for. Same job, different verdict, depending
            # only on the backend.
            if self.structured_outputs is None:
                raise RuntimeError(
                    "this vLLM exposes neither StructuredOutputsParams nor "
                    "GuidedDecodingParams, so response_format json_object "
                    "cannot be honoured on the offline backend"
                )
            keyword, params_cls = self.structured_outputs
            kwargs[keyword] = params_cls(json_object=True)
        return self.sampling_params_cls(**kwargs)

    def _response_for(self, output) -> _OfflineResponse:
        completion = output.outputs[0]
        choice = {
            "message": {"content": completion.text},
            "finish_reason": completion.finish_reason,
        }
        logprobs = self._openai_logprobs(completion)
        if logprobs is not None:
            choice["logprobs"] = logprobs
        return _OfflineResponse(
            {
                "choices": [choice],
                "usage": {
                    "prompt_tokens": len(output.prompt_token_ids or []),
                    "completion_tokens": len(completion.token_ids),
                },
            }
        )

    def _issue(self, payloads: list[dict[str, Any]],
               futures: list[cf.Future]) -> None:
        """Serve one template-homogeneous group, isolating whatever fails."""
        try:
            conversations = [payload["messages"] for payload in payloads]
            params = [self._sampling_params(payload) for payload in payloads]
            outputs = self.llm.chat(
                conversations,
                sampling_params=params,
                use_tqdm=False,
                **_chat_kwargs(payloads[0]),
            )
        except BaseException as exc:
            if len(payloads) == 1:
                _settle_exception(futures[0], exc)
                return
            # ONE BAD JOB MUST NOT FAIL ITS NEIGHBOURS. llm.chat() is
            # all-or-nothing, so a single unprocessable conversation (an
            # over-length prompt is the likely one) fails every future beside it
            # -- batch_size is --workers, 64 by default -- and retrying the same
            # group multiplies the cost by --retries. Reissuing one conversation
            # at a time costs one extra pass over the failed batch and narrows the
            # failure to the actual row.
            for payload, future in zip(payloads, futures):
                self._issue([payload], [future])
            return
        for output, future in zip(outputs, futures):
            try:
                response = self._response_for(output)
            except BaseException as exc:
                _settle_exception(future, exc)
                continue
            future.set_result(response)
        for future in futures:
            # Fewer outputs than conversations: a caller must get an error, not
            # a future that is never resolved.
            _settle_exception(
                future,
                RuntimeError("vLLM returned no output for this conversation"),
            )

    def _fail_outstanding(self) -> None:
        """Nobody drains this queue once the dispatcher is gone."""
        stopped = self.failure or RuntimeError("offline vLLM dispatcher stopped")
        while True:
            try:
                item = self.requests.get_nowait()
            except queue.Empty:
                return
            if item is not None:
                _settle_exception(item[1], stopped)

    def _run(self) -> None:
        try:
            while True:
                first = self.requests.get()
                if first is None:
                    return
                batch = [first]
                deadline = time.monotonic() + 0.002
                while len(batch) < self.batch_size:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        item = self.requests.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if item is None:
                        self.requests.put(None)
                        break
                    batch.append(item)

                # ONE llm.chat PER TEMPLATE GROUP. The chat template arguments
                # are per-CALL, so a batch may only hold conversations that agree
                # on them. Reading them from payloads[0] lets whichever request
                # arrives first inside the 2 ms window decide the thinking channel
                # and prefill geometry for every other job in the batch -- same
                # input, different verdict, by thread timing and unrecorded in the
                # output.
                groups: dict[tuple, list] = {}
                for item in batch:
                    groups.setdefault(_template_signature(item[0]), []).append(item)
                for group in groups.values():
                    self._issue([item[0] for item in group],
                                [item[1] for item in group])
        except BaseException as exc:
            self.failure = exc
            raise
        finally:
            self._fail_outstanding()

    def close(self) -> None:
        self.requests.put(None)
        self.worker.join()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()
