"""Run prepared grounding shards through vLLM server or offline inference.

The script keeps only two pieces of recovery machinery:

* each failed job is retried before being finalized with ``verdict="error"``;
* the final gzip JSON dictionary is written to a temporary file and atomically
  renamed, so interruption restarts only the unfinished shard and cannot
  publish a half-written result.

Examples
--------
Test the first 1,000 jobs of shard 0::

    PYTHONPATH=src python scripts/run_vllm_processed_shards.py \
      --shard-index 0 --limit 1000 --workers 64

Run one complete shard::

    PYTHONPATH=src python scripts/run_vllm_processed_shards.py \
      --shard-index 800 --workers 64

Run all shards; completed output shards are skipped::

    PYTHONPATH=src python scripts/run_vllm_processed_shards.py --workers 64

Load vLLM directly instead of using an HTTP server::

    PYTHONPATH=src python scripts/run_vllm_processed_shards.py \
      --backend offline --shard-index 800 --workers 96
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import gzip
import json
import pickle
import queue
import re
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_INPUT_DIR = Path("/scratch/h.yan/data/processed_grounding_shards")
DEFAULT_OUTPUT_DIR = Path("/scratch/h.yan/data/processed_model_results")
DEFAULT_GENE_OUTPUT_DIR = Path("/scratch/h.yan/data/processed_model_results_gene")
DEFAULT_MODEL = "vllm-gemma-4-26b"
# NO CoT at corpus scale. `verdict_only` is a coherent set — verdict-first
# prompt, thinking suppressed, temperature 0 — and all three must hold together:
# MEASURED, thinking off with the DELIBERATIVE prompt puts the verdict 56 tokens
# deep at delta_logit +22.50, indistinguishable from full deliberation and
# silently useless. Suppressing reasoning is therefore a property of the
# VARIANT, never a flag to flip on the reasoning-first path.
#
# The reasoning-first variant remains selectable for a gold-eval-sized run where
# its deliberation is affordable; at 60M evidences it is not.
DEFAULT_VARIANT = "verdict_only"
VALID_VERDICTS = {"correct", "incorrect"}
VALID_CONFIDENCE = {"high", "medium", "low"}
SHARD_RE = re.compile(r"grounded-(\d+)\.jsonl\.gz$")


def build_gene_stmt_hash_set() -> set[int]:
    """Save the hashes of all processed gene-only statements."""
    from indra.statements import stmt_from_json
    from indra.tools import assemble_corpus as ac
    from indra_db.readonly_dumping.locations import processed_stmts_fpath
    from indra_db.readonly_dumping.util import clean_json_loads
    from tqdm import tqdm

    batch_size = 100_000
    output_path = Path(processed_stmts_fpath).with_name("gene_stmt_hashes.pkl")
    gene_stmt_hashes: set[int] = set()
    batch_stmts: list[Any] = []
    batch_hashes: list[int] = []

    def filter_batch() -> None:
        filtered = ac.filter_genes_only(batch_stmts, specific_only=True)
        filtered_ids = {id(stmt) for stmt in filtered}
        gene_stmt_hashes.update(
            stmt_hash
            for stmt_hash, stmt in zip(batch_hashes, batch_stmts)
            if id(stmt) in filtered_ids
        )
        batch_stmts.clear()
        batch_hashes.clear()

    csv.field_size_limit(sys.maxsize)
    with gzip.open(processed_stmts_fpath, "rt") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for stmt_hash, stmt_json in tqdm(
            reader, total=48_769_436, unit="stmt", unit_scale=True
        ):
            stmt = stmt_from_json(clean_json_loads(stmt_json))
            batch_hashes.append(int(stmt_hash))
            batch_stmts.append(stmt)
            if len(batch_stmts) >= batch_size:
                filter_batch()

    if batch_stmts:
        filter_batch()
    with output_path.open("wb") as fh:
        pickle.dump(gene_stmt_hashes, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"gene statement hashes: {len(gene_stmt_hashes):,}")
    print(f"output: {output_path}")
    return gene_stmt_hashes


def iter_jobs(
    path: Path,
    limit: int | None = None,
    stmt_hashes: set[int] | None = None,
) -> Iterable[dict[str, Any]]:
    """Stream jobs from one gzip JSONL shard."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        count = 0
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                job = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if stmt_hashes is not None and int(job["stmt_hash"]) not in stmt_hashes:
                continue
            yield job
            count += 1
            if limit is not None and count >= limit:
                return


def job_id(job: dict[str, Any]) -> str:
    return str(job.get("job_id") or f"{job['input_row_index']}:0")


def valid_result(row: dict[str, Any] | None) -> bool:
    return bool(
        row
        and row.get("verdict") in VALID_VERDICTS
        and row.get("confidence") in VALID_CONFIDENCE
    )


def resolve_results_path(output_dir: Path, shard_index: int,
                        limit: int | None = None) -> Path | None:
    """Find the scored file for a shard, whatever --limit the run used.

    The canonical resolver, because reconstructing this name from a remembered
    --limit has now failed twice in two different consumers. Output names carry
    the limit (`verdicts-000000.limit-400.json.gz`), so a reader that rebuilds
    the unlimited name misses EVERY shard -- and both times the symptom was a
    clean exit over an empty result rather than an error.

    Exact match first; failing that, exactly one candidate for this shard index
    is accepted and anything ambiguous is refused by returning None, so a
    directory holding two generations of the same shard cannot be joined
    silently against the wrong one.
    """
    exact = output_path(output_dir, shard_index, limit)
    if exact.exists():
        return exact
    candidates = sorted(output_dir.glob(f"verdicts-{shard_index:06d}*.json.gz"))
    return candidates[0] if len(candidates) == 1 else None


def output_path(
    output_dir: Path,
    shard_index: int,
    limit: int | None,
) -> Path:
    tag = f"{shard_index:06d}"
    if limit is not None:
        tag += f".limit-{limit}"
    return output_dir / f"verdicts-{tag}.json.gz"


def write_final_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Publish a complete gzip JSON dictionary with one atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
        fh.write("\n")
    temporary.replace(path)


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


class OfflineVllmClient:
    """Batch concurrent post calls through one long-lived vllm.LLM."""

    def __init__(
        self,
        model_id: str,
        *,
        batch_size: int,
        gpu_memory_utilization: float,
    ):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise SystemExit(
                "offline backend requires vLLM: python -m pip install vllm"
            ) from exc

        self.sampling_params_cls = SamplingParams
        self.batch_size = batch_size
        self.requests: queue.Queue = queue.Queue()
        self.llm = LLM(
            model=model_id,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
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
        del timeout
        future: cf.Future = cf.Future()
        self.requests.put((json, future))
        return future.result()

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

    def _run(self) -> None:
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

            payloads = [item[0] for item in batch]
            futures = [item[1] for item in batch]
            conversations = [payload["messages"] for payload in payloads]
            params = [
                self.sampling_params_cls(
                    temperature=float(payload.get("temperature", 0.1)),
                    max_tokens=int(payload.get("max_tokens", 1000)),
                    logprobs=(
                        int(payload["top_logprobs"])
                        if payload.get("logprobs")
                        else None
                    ),
                )
                for payload in payloads
            ]
            chat_kwargs: dict[str, Any] = {}
            template_kwargs = payloads[0].get("chat_template_kwargs")
            if isinstance(template_kwargs, dict):
                chat_kwargs["chat_template_kwargs"] = template_kwargs
            try:
                outputs = self.llm.chat(
                    conversations,
                    sampling_params=params,
                    use_tqdm=False,
                    **chat_kwargs,
                )
                for output, future in zip(outputs, futures):
                    completion = output.outputs[0]
                    choice = {
                        "message": {"content": completion.text},
                        "finish_reason": completion.finish_reason,
                    }
                    logprobs = self._openai_logprobs(completion)
                    if logprobs is not None:
                        choice["logprobs"] = logprobs
                    future.set_result(
                        _OfflineResponse(
                            {
                                "choices": [choice],
                                "usage": {
                                    "prompt_tokens": len(
                                        output.prompt_token_ids or []
                                    ),
                                    "completion_tokens": len(completion.token_ids),
                                },
                            }
                        )
                    )
            except BaseException as exc:
                for future in futures:
                    future.set_exception(exc)

    def close(self) -> None:
        self.requests.put(None)
        self.worker.join()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()


class MonolithicPrompt:
    """Reuse the repo's commit-first disconfirm_relnature workflow."""

    # THREE SYMBOLS THIS CLASS WAS WRITTEN AGAINST NO LONGER EXIST, and the
    # replacements are not renames — they are the point of the K2/X1 unification
    # this branch predates by 48 commits:
    #
    #   parse_structured + derive_verdict -> indra_belief.verdict.parse_verdict
    #       One parser reads every reply, on the live path and the batch replay,
    #       under every profile. The retired pair also differed from the live
    #       reading under truncation, so hand-rolling them here would have made
    #       this runner a fourth reading of the corpus.
    #   _NATURE_LABEL -> prepared_execution.relation_mismatch_note
    #       The label table and the mismatch sentence have one owner; a byte
    #       change there must move the live and batch prompts together, which a
    #       local copy in this file would silently break.
    #
    # The class's own docstring says "reuse the repo's workflow". On this tree
    # that means delegating to those owners rather than importing their parts.
    def __init__(self, variant: str = DEFAULT_VARIANT):
        from indra_belief.scorers.monolithic import scorer as mono
        from indra_belief.scorers.monolithic._prompts_relation import (
            _RELATION_SYSTEM,
            _extract_json,
            _norm_nature,
            _user_message,
        )
        from indra_belief.prepared_execution import relation_mismatch_note
        from indra_belief.verdict import parse_verdict

        # WAS a pinned import of DISCONFIRM_SYSTEM_PROMPT. Pinning made the
        # runner structurally incapable of the no-CoT path: a variant is a
        # COHERENT SET — verdict-first prompt, thinking suppressed, temperature
        # 0 — and a runner that borrowed one member of that set and hard-coded
        # the rest could only ever send an inconsistent request. Reading the
        # whole set off one object is what keeps the three from drifting.
        try:
            self.variant = mono.VARIANTS[variant]
        except KeyError:
            raise SystemExit(
                f"unknown --variant {variant!r}; registered: "
                f"{', '.join(sorted(n for n in mono.VARIANTS if n))}"
            ) from None

        self.mono = mono
        self.system_prompt = self.variant.system_prompt
        self.render_example = self.variant.render_example
        self.relation_system_prompt = _RELATION_SYSTEM
        self.extract_json = _extract_json
        self.normalize_nature = _norm_nature
        self.relation_user_message = _user_message
        self.relation_mismatch_note = relation_mismatch_note
        self.parse_verdict = parse_verdict

    @lru_cache(maxsize=None)
    def examples(self, stmt_type: str) -> tuple[dict[str, str], ...]:
        messages: list[dict[str, str]] = []
        for example in self.mono._select_examples(stmt_type):
            user, assistant = self.render_example(example)
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
        return tuple(messages)

    def relation_request(
        self, job: dict[str, Any]
    ) -> tuple[str, list[dict[str, str]]] | None:
        """Build the extra relation-nature call used only for Complex jobs.

        Gated on the VARIANT, not just the statement type: a variant with no
        relation resolver (``verdict_only``) must not pay for a second call it
        has no prompt to consume, and the live scorer already skips it for
        exactly that reason.
        """
        if self.variant.resolve_relation_nature is None:
            return None
        if str(job.get("stmt_type")) != "Complex":
            return None
        subject = str(job.get("subject") or "")
        object_ = str(job.get("object") or "")
        evidence = str(job.get("evidence_text") or "")
        if not subject or not object_ or subject == "?" or object_ == "?" or not evidence:
            return None
        user = self.relation_user_message(
            subject,
            object_,
            evidence,
            job.get("subject_grounding"),
            job.get("object_grounding"),
        )
        return self.relation_system_prompt, [{"role": "user", "content": user}]

    def relation_note(
        self, job: dict[str, Any], content: str, reasoning: str
    ) -> str:
        """Turn a non-binding Complex classification into a rejection note."""
        obj = self.extract_json(content)
        if not isinstance(obj, dict) and reasoning:
            obj = self.extract_json(reasoning)
        if not isinstance(obj, dict):
            return ""
        # The parsing and normalization stay here — they are path-specific, and
        # the live and batch `_relation_note` wrappers keep their own for the
        # same reason. Only the RENDERING delegates, because that is the twin.
        return self.relation_mismatch_note(
            self.normalize_nature(obj.get("nature")),
            obj.get("span") or "",
            job["subject"],
            job["object"],
        )

    def request(
        self, job: dict[str, Any], relation_note: str = ""
    ) -> tuple[str, list[dict[str, str]]]:
        user_message = job.get("user_message")
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("LLM job has no user_message")
        if relation_note:
            user_message += "\n\n" + relation_note
        system = self.system_prompt
        if job.get("lookup_guidance_required"):
            system += self.mono._LOOKUP_GUIDANCE
        messages = list(self.examples(str(job.get("stmt_type") or "Unknown")))
        messages.append({"role": "user", "content": user_message})
        return system, messages

    def parse(self, content: str, reasoning: str) -> tuple[str | None, str | None]:
        """Read a reply through the ONE parser, in the live path's own order.

        `indra_belief.verdict` is the single reader for every path. The two-text
        attempt is kept because it is this runner's own contract with a local
        model that may put the answer in either channel; what is no longer local
        is the READING, so a reply that scores here scores identically on the
        live scorer and the batch replay.

        An unreadable reply stays (None, None). Parsing never fabricates a
        probability from categorical output.
        """
        for text in ([content, f"{reasoning}\n{content}"] if reasoning else [content]):
            read = self.parse_verdict(text)
            if read is not None and read.label is not None:
                return read.label, read.confidence
        return None, None


def score_job(
    job: dict[str, Any],
    *,
    client,
    prompt: MonolithicPrompt,
    endpoint: str,
    model_id: str,
    max_tokens: int,
    temperature: float,
    probe: bool = False,
) -> dict[str, Any]:
    """Return the minimal row needed for resume and finalization.

    ``probe`` adds ONE forced-position label read per scored evidence and
    persists its raw ``probe_delta_logit``. Off by default because at corpus
    scale it doubles the request count; on for a gold-eval-sized run it is
    minutes, and it is the only way a serving stack collects the data needed to
    fit its own calibration without a second pass.

    The WIDTH is deliberately not a parameter — see ``_read_probe_delta``.
    """
    base = {
        "job_id": job_id(job),
        "stmt_hash": str(job["stmt_hash"]),
        "source_hash": str(job["source_hash"]),
    }

    if not job.get("needs_llm", True):
        result = job.get("tier1_result") or {}
        if valid_result(result):
            return {
                **base,
                "verdict": result["verdict"],
                "confidence": result["confidence"],
                "source": "tier1",
            }
        return {**base, "verdict": None, "confidence": None, "error": "bad tier1"}

    relation_note = ""
    try:
        relation_builder = getattr(prompt, "relation_request", None)
        relation_request = relation_builder(job) if relation_builder else None
        if relation_request is not None:
            relation_system, relation_messages = relation_request
            relation_response = client.post(
                endpoint,
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": relation_system}
                    ]
                    + relation_messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                },
            )
            relation_response.raise_for_status()
            relation_payload = relation_response.json()
            relation_message = relation_payload["choices"][0].get("message") or {}
            relation_content = str(relation_message.get("content") or "")
            relation_reasoning = str(
                relation_message.get("reasoning_content")
                or relation_message.get("reasoning")
                or ""
            )
            relation_note = prompt.relation_note(
                job, relation_content, relation_reasoning
            )
    except Exception:
        # Match the monolithic scorer: an unavailable or unparseable focused
        # step leaves the holistic Complex verdict untouched.
        relation_note = ""

    try:
        if relation_note:
            system, messages = prompt.request(job, relation_note)
        else:
            system, messages = prompt.request(job)
    except Exception as exc:
        return {
            **base,
            "verdict": None,
            "confidence": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    variant = prompt.variant
    # The variant's own declarations, not this runner's defaults. Reasoning is
    # the one that used to be MISSING ENTIRELY: the body below carried model,
    # messages, max_tokens and temperature, and nothing else — so the thinking
    # channel was whatever the served chat template happened to default to. On
    # gemma-4 that is ON, and at 60M evidences an unasked-for CoT is the entire
    # bill.
    #
    # `reasoning_wire_keys` rather than a literal, because "no CoT" is not one
    # key: vLLM/Ollama-served Gemma silently DROPS `reasoning_effort="none"` and
    # honors `chat_template_kwargs.enable_thinking` instead. Sending one of the
    # two is a silent no-op on half the substrates.
    from indra_belief.model_client import reasoning_wire_keys

    body = {
        "model": model_id,
        "messages": [{"role": "system", "content": system}] + messages,
        "max_tokens": max_tokens,
        "temperature": (
            variant.temperature if variant.temperature is not None else temperature
        ),
        **reasoning_wire_keys(variant.reasoning_effort),
    }
    if variant.in_call_label_logprobs:
        # Free margin: this variant emits the verdict FIRST, so the label's
        # log-odds are readable from the call we were making anyway. MEASURED
        # n=80: in-call AUROC 0.8734 against the second-call probe's 0.7237 —
        # better AND without doubling the request count.
        body["logprobs"] = True
        body["top_logprobs"] = variant.in_call_label_logprobs
        if body["temperature"] != 0:
            # The same invariant ModelClient enforces. Above temperature 0 the
            # reported argmax stream can diverge from the sampled text, so the
            # verdict POSITION stops being trustworthy and the margin is a
            # plausible-looking lie. A CLI --temperature must not be able to
            # reach past the variant and break this quietly.
            raise SystemExit(
                f"variant {variant.name!r} reads in-call logprobs but temperature "
                f"is {body['temperature']}; that read is only valid at 0"
            )
    try:
        response = client.post(endpoint, json=body)
        response.raise_for_status()
        payload = response.json()
        choice = payload["choices"][0]
        message = choice.get("message") or {}
        content = str(message.get("content") or "")
        reasoning = str(
            message.get("reasoning_content") or message.get("reasoning") or ""
        )
        verdict, confidence = prompt.parse(content, reasoning)
        if verdict in VALID_VERDICTS and confidence in VALID_CONFIDENCE:
            row = {
                **base,
                "verdict": verdict,
                "confidence": confidence,
                "source": "llm",
            }
            # Free path first, exactly as the live scorer resolves it. When the
            # variant emits the verdict first, the margin is already in THIS
            # response; issuing `--probe`'s second request on top would pay a
            # doubled request count for a strictly worse reading (n=80: in-call
            # AUROC 0.8734, probe 0.7237).
            if variant.in_call_label_logprobs:
                from indra_belief.probes.reader import label_margin_from_payload

                row["probe_delta_logit"] = label_margin_from_payload(payload)
            elif probe:
                row["probe_delta_logit"] = _read_probe_delta(
                    client, endpoint, model_id, job
                )
            return row
        return {
            **base,
            "verdict": None,
            "confidence": None,
            "error": "unparseable model response",
            "finish_reason": choice.get("finish_reason"),
            "completion_tokens": (payload.get("usage") or {}).get(
                "completion_tokens"
            ),
            # Diagnostic only: store at most 4,000 characters of the model's
            # response object, never the input prompt.
            "response_preview": json.dumps(
                message, ensure_ascii=False, default=str
            )[:4000],
        }
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {**base, "verdict": None, "confidence": None, "error": error}



def _read_probe_delta(client, endpoint, model_id, job):
    """The forced-position label read, over this runner's own httpx client.

    Uses the shared request builder and parser rather than hand-rolling the
    body: three things make the read work (thinking suppressed, an assistant
    prefill opening the JSON string, `continue_final_message` extending that
    turn) and getting any one wrong yields a SATURATED number rather than an
    error. MEASURED: thinking off but with the production prompt puts the
    verdict 56 tokens deep at delta_logit +22.50 — indistinguishable from full
    deliberation, and silently useless.

    Returns the RAW log-odds or None. Never a probability: without a fitted
    isotonic for this serving stack there is no calibration, and the value is
    comparable only within one stack. Persisting it is what lets a stack fit its
    own calibration offline from a run it already did, instead of needing a
    second pass over the corpus.

    A failure here must not lose the verdict — the probe is an extra
    measurement, not a precondition — so every error degrades to None.
    """
    from indra_belief.probes.reader import (
        PROBE_FIRST_TRY_TOP_LOGPROBS,
        PROBE_TOP_LOGPROBS,
        ProbeTopKError,
        build_probe_request,
        probe_reading_from_payload,
    )

    # The width is NOT a caller knob. It is measured (losing-label rank median 6,
    # max 15 over 40 records on MLX) and widens on demand, so every caller reads
    # the same number and a stack that ranks differently is discovered by
    # probe_widen_count() rather than papered over by a command-line guess.
    top_logprobs = PROBE_FIRST_TRY_TOP_LOGPROBS

    record = {
        "subject": job.get("subject"),
        "object": job.get("object"),
        "stmt_type": job.get("stmt_type"),
        "evidence_text": job.get("evidence_text") or "",
    }

    def _issue(width: int) -> float:
        body = build_probe_request(
            record, model_id=model_id, top_logprobs=width, inline_extra_body=True
        )
        response = client.post(endpoint, json=body)
        response.raise_for_status()
        return probe_reading_from_payload(response.json(), top_k=width).delta_logit

    try:
        return _issue(top_logprobs)
    except ProbeTopKError:
        # Same widen-on-demand as read_probe. Without it this path SILENTLY
        # dropped any record whose losing label sat outside the window — and the
        # width is measured on MLX, so a stack that ranks differently would lose
        # readings with nothing to show for it. Widening keeps the parity: one
        # extra call on a rare record, never a lost measurement.
        ceiling = min(PROBE_TOP_LOGPROBS, max(top_logprobs, PROBE_TOP_LOGPROBS))
        if ceiling <= top_logprobs:
            return None
        try:
            return _issue(ceiling)
        except Exception:
            return None
    except Exception:
        return None


def score_job_with_retries(
    job: dict[str, Any],
    *,
    retries: int,
    **score_kwargs: Any,
) -> dict[str, Any]:
    """Score one job, retrying failures up to the requested additional times."""
    attempts = retries + 1 if job.get("needs_llm", True) else 1
    row: dict[str, Any] | None = None
    for _attempt in range(1, attempts + 1):
        row = score_job(job, **score_kwargs)
        if valid_result(row):
            return row
    assert row is not None
    row["attempts"] = attempts
    return row


def finalize(
    input_path: Path,
    latest: dict[str, dict[str, Any]],
    limit: int | None,
    stmt_hashes: set[int] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Build the hash dictionary, retaining exhausted failures as errors."""
    payload: dict[str, dict[str, dict[str, Any]]] = {}
    for job in iter_jobs(input_path, limit, stmt_hashes):
        result = latest.get(job_id(job))
        if result is None:
            raise RuntimeError(f"job produced no result: {job_id(job)}")
        stmt_hash = str(job["stmt_hash"])
        source_hash = str(job["source_hash"])
        if valid_result(result):
            cell: dict[str, Any] = {
                "verdict": str(result["verdict"]),
                "confidence": str(result["confidence"]),
            }
            if result.get("probe_delta_logit") is not None:
                cell["probe_delta_logit"] = float(result["probe_delta_logit"])
        else:
            cell = {
                "verdict": "error",
                "confidence": None,
                "error": str(result.get("error") or "unknown error"),
                "attempts": int(result.get("attempts") or 1),
            }
            for key in ("finish_reason", "completion_tokens", "response_preview"):
                if key in result:
                    cell[key] = result[key]
        payload.setdefault(stmt_hash, {})[source_hash] = cell
    return payload


def run_shard(
    input_path: Path,
    args,
    client,
    prompt: MonolithicPrompt,
    stmt_hashes: set[int] | None = None,
) -> int:
    from tqdm import tqdm

    match = SHARD_RE.search(input_path.name)
    assert match
    shard_index = int(match.group(1))
    output_dir = Path(args.output_dir)
    final_path = output_path(output_dir, shard_index, args.limit)

    if final_path.exists():
        print(f"skip completed shard {shard_index}: {final_path}")
        return 0

    total = sum(1 for _ in iter_jobs(input_path, args.limit, stmt_hashes))
    latest: dict[str, dict[str, Any]] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    pending = iter_jobs(input_path, args.limit, stmt_hashes)
    started = time.perf_counter()
    errors = 0
    llm = 0
    tier1 = 0
    endpoint = args.base_url.rstrip("/") + "/chat/completions"

    print(
        f"shard={shard_index} jobs={total:,} workers={args.workers} "
        f"retries={args.retries}"
    )
    progress = tqdm(total=total, desc="Scoring", unit="job")

    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        inflight: set[cf.Future] = set()

        def submit_one() -> bool:
            try:
                job = next(pending)
            except StopIteration:
                return False
            inflight.add(
                pool.submit(
                    score_job_with_retries,
                    job,
                    retries=args.retries,
                    client=client,
                    probe=bool(getattr(args, "probe", False)),
                    prompt=prompt,
                    endpoint=endpoint,
                    model_id=args.model_id,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
            )
            return True

        for _ in range(args.workers * 2):
            if not submit_one():
                break

        while inflight:
            finished, _ = cf.wait(inflight, return_when=cf.FIRST_COMPLETED)
            for future in finished:
                inflight.remove(future)
                row = future.result()
                latest[str(row["job_id"])] = row
                errors += int(not valid_result(row))
                llm += int(row.get("source") == "llm")
                tier1 += int(row.get("source") == "tier1")
                progress.update(1)
                progress.set_postfix(llm=llm, tier1=tier1, errors=errors)
                submit_one()

    progress.close()
    elapsed = time.perf_counter() - started
    payload = finalize(input_path, latest, args.limit, stmt_hashes)
    write_final_atomic(final_path, payload)
    evidence_results = sum(len(by_source) for by_source in payload.values())
    print(
        f"completed shard {shard_index}: {evidence_results:,} evidence results "
        f"for {len(payload):,} statements in {elapsed / 60:.2f} minutes "
        f"({evidence_results / max(elapsed, 1e-9):.2f} jobs/s), "
        f"errors={errors:,}"
    )
    print(f"output={final_path}")
    return 0


def select_shards(input_dir: Path, shard_index: int | None) -> list[Path]:
    if shard_index is not None:
        path = input_dir / f"grounded-{shard_index:06d}.jsonl.gz"
        if not path.exists():
            raise SystemExit(f"input shard does not exist: {path}")
        return [path]
    paths = sorted(input_dir.glob("grounded-*.jsonl.gz"))
    if not paths:
        raise SystemExit(f"no shards found in {input_dir}")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument(
        "--output-dir",
        help="Defaults to processed_model_results, or processed_model_results_gene "
        "when --gene-stmt-hashes is used.",
    )
    parser.add_argument(
        "--gene-stmt-hashes",
        help="Pickle file containing the set of statement hashes to process.",
    )
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--backend",
        choices=("server", "offline"),
        default="server",
        help="Use the existing HTTP server or load vLLM in this process.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--variant", default=DEFAULT_VARIANT,
        help=(
            "scoring variant; carries its prompt, reasoning channel and "
            f"temperature as one set (default {DEFAULT_VARIANT}: no CoT, and "
            "the label margin comes free from the scoring call)"
        ),
    )
    parser.add_argument("--probe", action="store_true",
                        help="read the forced-position label logits per evidence and "
                             "persist the raw probe_delta_logit. Costs ONE extra "
                             "request per evidence — minutes at gold-eval scale, a "
                             "doubling of request count at 60M, which is why it is "
                             "opt-in. The window is NOT a knob: it comes from the "
                             "measured PROBE_FIRST_TRY_TOP_LOGPROBS and widens on "
                             "demand, so one definition governs every caller.")
    parser.add_argument("--require-calibrated", action="store_true",
                        help="refuse the run unless this model+prompt resolves a "
                             "ship-approved calibration profile")
    parser.add_argument("--base-url", help="Override the model registry URL.")
    parser.add_argument(
        "--served-model-id",
        help="Override the registry model ID (server name or offline model path).",
    )
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Additional attempts after a failed LLM result (default: 3).",
    )
    # max-tokens and timeout DEFAULT TO THE REGISTRY, matching
    # scripts/run_rasmachine_monolithic.py. They used to be hardcoded here at
    # 1000/180, which silently overrode the registry entry for every run: a
    # 1000-token cap truncates 16.7% of calls under the production
    # reasoning-first prompt (measured, n=60: p50 574, p90 1507, max 4353), and
    # a truncated read costs the full wall clock while yielding no verdict.
    # A ceiling belongs with the model, not with one of its callers.
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM offline backend GPU-memory fraction (default: 0.9).",
    )
    return parser


def preflight(client, endpoint: str, model_id: str, prompt) -> None:
    """One request, before any shard is opened, to prove the server agrees.

    WHY THIS IS NOT OPTIONAL POLISH
    -------------------------------
    The no-CoT variant reads the label margin from the scoring call, which means
    every request carries `logprobs: true` and a 128-wide window. vLLM caps that
    at `--max-logprobs`, whose default is far below 128 — so a server started
    without the flag REJECTS EVERY REQUEST. Not the margin: the whole call,
    because the verdict rides in the same response. A 60M-evidence run would
    fail every row for a missing server flag, and the operator would find out
    one shard in.

    That is the exact failure this file's own comments say must not happen —
    "the margin is an extra measurement, not a precondition". At the transport
    level it IS a precondition, and the honest fix is to discover it in two
    seconds against one request rather than degrade silently across a corpus.

    Fails loudly with the remedy in the message. Never silently drops the
    logprob request: a run that quietly stopped collecting margins would look
    exactly like a successful one.
    """
    variant = prompt.variant
    from indra_belief.model_client import reasoning_wire_keys

    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Reply with the single word ok."}],
        "max_tokens": 4,
        "temperature": variant.temperature if variant.temperature is not None else 0.0,
        **reasoning_wire_keys(variant.reasoning_effort),
    }
    if variant.in_call_label_logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = variant.in_call_label_logprobs

    try:
        response = client.post(endpoint, json=body, timeout=120)
    except Exception as exc:
        raise SystemExit(
            f"[preflight] cannot reach {endpoint}: {type(exc).__name__}: {exc}\n"
            "  is the server up, and does --base-url point at it?"
        ) from None

    if response.status_code >= 400:
        detail = response.text[:400]
        hint = ""
        if variant.in_call_label_logprobs and "logprob" in detail.lower():
            hint = (
                f"\n  the server refused a {variant.in_call_label_logprobs}-wide "
                "logprob window. Restart it with\n"
                f"      --max-logprobs {variant.in_call_label_logprobs}\n"
                "  (vLLM's default is well below that). Every request carries this "
                "window because\n  the verdict and its margin come from the same "
                "call, so this rejects the whole run."
            )
        raise SystemExit(
            f"[preflight] server returned {response.status_code}: {detail}{hint}"
        )

    if variant.in_call_label_logprobs:
        from indra_belief.probes.reader import label_margin_from_payload

        choices = (response.json() or {}).get("choices") or []
        got = bool(choices) and bool((choices[0] or {}).get("logprobs"))
        if not got:
            raise SystemExit(
                "[preflight] the server accepted the request but returned NO "
                "logprobs.\n  Every row would carry a null margin while looking "
                "perfectly healthy.\n  Check that the serving stack supports "
                "top_logprobs on chat completions."
            )
        # A 'logprobs came back' check is NOT enough, and this used to stop
        # there. The reader locates the margin by the EMITTED TOKEN matching a
        # label, so a tokenizer that emits " correct" with a leading space, or
        # "correct\"" with the quote attached, returns None -- silently, per row,
        # forever. Verdicts still land, the run looks healthy, and 60M margins
        # are null. MEASURED: both variants return None today.
        #
        # So the preflight issues a REAL SCORING REQUEST and parses it with the
        # function the runner uses, `label_margin_from_payload`.
        #
        # An earlier version of this check issued `build_probe_request` instead
        # and claimed it was "parsed the same way". It is not, and the
        # difference is exactly the hazard: the probe route forces the label to
        # generated position 0 with a prefill and looks it up among that
        # position's alternatives, so it CANNOT exhibit the emitted-token
        # mismatch. The in-call route scans for the first position whose EMITTED
        # token is a label, which is what a leading space or an attached quote
        # breaks. The check validated the one route that could not fail and left
        # the one that does unprotected.
        job = {
            "stmt_type": "Activation",
            "user_message": "CLAIM: A [Activation] B\nEVIDENCE: A activates B.",
        }
        try:
            system, messages = prompt.request(job)
            probe_body = {
                "model": model_id,
                "messages": [{"role": "system", "content": system}] + messages,
                "max_tokens": 64,
                "temperature": body["temperature"],
                "logprobs": True,
                "top_logprobs": variant.in_call_label_logprobs,
                **reasoning_wire_keys(variant.reasoning_effort),
            }
            probe_response = client.post(endpoint, json=probe_body, timeout=180)
            probe_response.raise_for_status()
            margin = label_margin_from_payload(probe_response.json())
        except Exception as exc:
            raise SystemExit(
                f"[preflight] a real scoring request failed: "
                f"{type(exc).__name__}: {exc}"
            ) from None
        if margin is None:
            raise SystemExit(
                "[preflight] a real scoring call returned logprobs, but NO label "
                "margin could be read from it.\n  That is what a tokenizer "
                "mismatch looks like: the reader locates the margin by the\n"
                "  EMITTED token matching a label, and a leading space or an "
                "attached quote yields None.\n  Verdicts would still land, every "
                "margin would be null, and the run would look healthy\n  until "
                "the calibration fit had nothing to fit on."
            )
        print(f"[preflight] in-call label margin readable on this stack "
              f"(delta_logit {margin:+.3f}) — the same read the run uses",
              flush=True)
    print(
        f"[preflight] ok — server accepts variant {variant.name!r} "
        f"(reasoning={variant.reasoning_effort or 'default'}, "
        f"logprobs={variant.in_call_label_logprobs or 'off'})",
        flush=True,
    )


def main() -> int:
    args = build_parser().parse_args()
    if args.workers < 1 or args.retries < 0:
        raise SystemExit("workers must be positive and retries nonnegative")
    if args.max_tokens is not None and args.max_tokens < 1:
        raise SystemExit("max-tokens must be positive")
    if args.timeout is not None and args.timeout <= 0:
        raise SystemExit("timeout must be positive")
    if args.limit is not None and (args.limit < 1 or args.shard_index is None):
        raise SystemExit("--limit must be positive and used with --shard-index")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise SystemExit("--gpu-memory-utilization must be in (0, 1]")

    from indra_belief.model_client import LOCAL_MODELS, canonical_model_name

    # CANONICALISE FIRST. The registry resolves aliases and this lookup did not,
    # so a renamed entry -- `vllm-local` -> `vllm-gemma-4-26b` -- would reject a
    # collaborator's existing command line with "unknown model registry entry"
    # while the alias it was renamed under resolved perfectly everywhere else.
    # The calibration profile is keyed on the CANONICAL name too, so resolving
    # here is what makes an aliased run and a canonical run the same run.
    args.model = canonical_model_name(args.model)
    if args.model not in LOCAL_MODELS:
        raise SystemExit(f"unknown model registry entry: {args.model}")
    model_config = LOCAL_MODELS[args.model]
    args.base_url = (args.base_url or model_config["base_url"]).rstrip("/")
    args.model_id = args.served_model_id or model_config["model_id"]
    # Registry is the source of truth for the ceiling; an explicit flag still wins.
    if args.max_tokens is None:
        args.max_tokens = model_config.get("max_tokens")
        if args.max_tokens is None:
            raise SystemExit(
                f"registry entry {args.model!r} declares no max_tokens; "
                "pass --max-tokens explicitly"
            )
    if args.timeout is None:
        args.timeout = float(model_config.get("timeout") or 900)

    stmt_hashes: set[int] | None = None
    if args.gene_stmt_hashes:
        gene_hash_path = Path(args.gene_stmt_hashes)
        with gene_hash_path.open("rb") as fh:
            loaded_hashes = pickle.load(fh)
        if not isinstance(loaded_hashes, set):
            raise SystemExit(f"gene hash pickle must contain a set: {gene_hash_path}")
        stmt_hashes = {int(value) for value in loaded_hashes}
        print(f"gene_stmt_hashes={gene_hash_path} hashes={len(stmt_hashes):,}")

    if args.output_dir is None:
        args.output_dir = str(
            DEFAULT_GENE_OUTPUT_DIR if stmt_hashes is not None else DEFAULT_OUTPUT_DIR
        )
    # Built HERE, before the [config] line and the banner that both read it.
    # An unknown --variant must fail before any shard is opened, and both of
    # those lines report properties OF this object — deferring construction is
    # how the previous version of this function came to raise NameError on
    # every invocation.
    prompt = MonolithicPrompt(args.variant)
    print(
        f"[config] model={args.model} served_id={args.model_id} "
        f"max_tokens={args.max_tokens} timeout={args.timeout:g}s "
        f"workers={args.workers} temperature={args.temperature} "
        f"variant={args.variant} reasoning={prompt.variant.reasoning_effort or 'default'}",
        flush=True,
    )

    # A calibration profile is keyed on (model, prompt sha), so the prompt this
    # path actually sends decides whether its beliefs are calibrated — and an
    # unfitted pair falls back to the hard gate SILENTLY. At 60M statements that
    # is the difference between ECE 0.045 and ECE 0.237 with nothing downstream
    # able to tell which it got.
    #
    # Hashed off the CONSTRUCTED PROMPT OBJECT, not a re-import of one variant's
    # constant. While the runner pinned DISCONFIRM_SYSTEM_PROMPT the two were the
    # same string; the moment --variant could change what is sent, a re-import
    # would have reported the calibration status of a prompt this run never
    # used — a banner that is confidently wrong is worse than no banner.
    import hashlib

    from indra_belief.calibration_constants import calibration_banner

    prompt_sha256 = hashlib.sha256(prompt.system_prompt.encode("utf-8")).hexdigest()
    calibrated, banner = calibration_banner(args.model, prompt_sha256)
    print(banner, flush=True)
    if args.require_calibrated and not calibrated:
        raise SystemExit(
            "refusing to run: --require-calibrated was passed and this "
            "model+prompt pair has no ship-approved profile"
        )

    shards = select_shards(Path(args.input_dir), args.shard_index)
    if args.backend == "offline":
        client_context = OfflineVllmClient(
            args.model_id,
            batch_size=args.workers,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    else:
        try:
            import httpx
        except ImportError as exc:
            raise SystemExit(
                "server backend requires httpx: python -m pip install httpx"
            ) from exc
        limits = httpx.Limits(
            max_connections=args.workers,
            max_keepalive_connections=args.workers,
        )
        client_context = httpx.Client(limits=limits, timeout=args.timeout)

    with client_context as client:
        preflight(client, args.base_url.rstrip("/") + "/chat/completions",
                  args.model_id, prompt)
        for shard in shards:
            code = run_shard(shard, args, client, prompt, stmt_hashes)
            if code:
                return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
