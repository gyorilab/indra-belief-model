"""Run prepared grounding shards through a running vLLM server.

The script keeps only two pieces of recovery machinery:

* completed jobs are appended to ``.partial.jsonl`` and skipped after restart;
* the final gzip JSON dictionary is written to a temporary file and atomically
  renamed, so interruption cannot publish a half-written result.

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
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import json
import os
import re
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_INPUT_DIR = Path("/scratch/h.yan/data/processed_grounding_shards")
DEFAULT_OUTPUT_DIR = Path("/scratch/h.yan/data/processed_model_results")
DEFAULT_MODEL = "vllm-local"
VALID_VERDICTS = {"correct", "incorrect"}
VALID_CONFIDENCE = {"high", "medium", "low"}
SHARD_RE = re.compile(r"grounded-(\d+)\.jsonl\.gz$")


def iter_jobs(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    """Stream jobs from one gzip JSONL shard."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        count = 0
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
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


def output_paths(
    output_dir: Path,
    shard_index: int,
    limit: int | None,
) -> tuple[Path, Path]:
    tag = f"{shard_index:06d}"
    if limit is not None:
        tag += f".limit-{limit}"
    final_path = output_dir / f"verdicts-{tag}.json.gz"
    partial_path = output_dir / f".verdicts-{tag}.partial.jsonl"
    return final_path, partial_path


def load_partial(path: Path) -> dict[str, dict[str, Any]]:
    """Load the latest attempt for each job, ignoring a truncated last line."""
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
                latest[str(row["job_id"])] = row
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return latest


def ensure_append_boundary(path: Path) -> None:
    """Separate a crash-truncated JSON fragment from newly appended rows."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb") as fh:
        fh.seek(-1, os.SEEK_END)
        ends_with_newline = fh.read(1) == b"\n"
    if not ends_with_newline:
        with path.open("ab") as fh:
            fh.write(b"\n")


def write_final_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Publish a complete gzip JSON dictionary with one atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
        fh.write("\n")
    temporary.replace(path)


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
    def __init__(self):
        from indra_belief.scorers.monolithic import scorer as mono
        from indra_belief.scorers.monolithic._prompts_disconfirm import (
            DISCONFIRM_SYSTEM_PROMPT,
            render_example,
        )
        from indra_belief.scorers.monolithic._prompts_relation import (
            _RELATION_SYSTEM,
            _extract_json,
            _norm_nature,
            _user_message,
        )
        from indra_belief.prepared_execution import relation_mismatch_note
        from indra_belief.verdict import parse_verdict

        self.mono = mono
        self.system_prompt = DISCONFIRM_SYSTEM_PROMPT
        self.relation_system_prompt = _RELATION_SYSTEM
        self.render_example = render_example
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
        """Build the extra relation-nature call used only for Complex jobs."""
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
) -> dict[str, Any]:
    """Return the minimal row needed for resume and finalization."""
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

    try:
        response = client.post(
            endpoint,
            json={
                "model": model_id,
                "messages": [{"role": "system", "content": system}] + messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
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
            return {
                **base,
                "verdict": verdict,
                "confidence": confidence,
                "source": "llm",
            }
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


def finalize(
    input_path: Path,
    latest: dict[str, dict[str, Any]],
    limit: int | None,
) -> tuple[dict[str, dict[str, dict[str, str]]], list[str]]:
    """Build ``{stmt_hash: {source_hash: {verdict, confidence}}}``."""
    payload: dict[str, dict[str, dict[str, str]]] = {}
    missing: list[str] = []
    for job in iter_jobs(input_path, limit):
        result = latest.get(job_id(job))
        if not valid_result(result):
            missing.append(job_id(job))
            continue
        stmt_hash = str(job["stmt_hash"])
        source_hash = str(job["source_hash"])
        payload.setdefault(stmt_hash, {})[source_hash] = {
            "verdict": str(result["verdict"]),
            "confidence": str(result["confidence"]),
        }
    return payload, missing


def run_shard(input_path: Path, args, client, prompt: MonolithicPrompt) -> int:
    from tqdm import tqdm

    match = SHARD_RE.search(input_path.name)
    assert match
    shard_index = int(match.group(1))
    output_dir = Path(args.output_dir)
    final_path, partial_path = output_paths(output_dir, shard_index, args.limit)

    if final_path.exists():
        print(f"skip completed shard {shard_index}: {final_path}")
        return 0

    total = sum(1 for _ in iter_jobs(input_path, args.limit))
    latest = load_partial(partial_path)
    done = {key for key, row in latest.items() if valid_result(row)}
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_append_boundary(partial_path)

    pending = (job for job in iter_jobs(input_path, args.limit) if job_id(job) not in done)
    started = time.perf_counter()
    errors = 0
    llm = 0
    tier1 = 0
    endpoint = args.base_url.rstrip("/") + "/chat/completions"

    print(
        f"shard={shard_index} jobs={total:,} resumed={len(done):,} "
        f"workers={args.workers}"
    )
    progress = tqdm(total=total, initial=len(done), desc="Scoring", unit="job")

    with partial_path.open("a", buffering=1) as partial_fh:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            inflight: set[cf.Future] = set()

            def submit_one() -> bool:
                try:
                    job = next(pending)
                except StopIteration:
                    return False
                inflight.add(
                    pool.submit(
                        score_job,
                        job,
                        client=client,
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
                    partial_fh.write(json.dumps(row) + "\n")
                    partial_fh.flush()
                    latest[str(row["job_id"])] = row
                    errors += int(not valid_result(row))
                    llm += int(row.get("source") == "llm")
                    tier1 += int(row.get("source") == "tier1")
                    progress.update(1)
                    progress.set_postfix(llm=llm, tier1=tier1, errors=errors)
                    submit_one()

    progress.close()
    elapsed = time.perf_counter() - started
    payload, missing = finalize(input_path, latest, args.limit)
    if missing:
        print(
            f"shard {shard_index} has {len(missing):,} failed jobs; "
            "rerun the same command to retry them"
        )
        return 2

    write_final_atomic(final_path, payload)
    partial_path.unlink(missing_ok=True)
    evidence_results = sum(len(by_source) for by_source in payload.values())
    print(
        f"completed shard {shard_index}: {evidence_results:,} evidence results "
        f"for {len(payload):,} statements in {elapsed / 60:.2f} minutes "
        f"({evidence_results / max(elapsed, 1e-9):.2f} jobs/s)"
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
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", help="Override the model registry URL.")
    parser.add_argument(
        "--served-model-id",
        help="Override the model name accepted by the vLLM server.",
    )
    parser.add_argument("--workers", type=int, default=64)
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.workers < 1:
        raise SystemExit("workers must be positive")
    if args.max_tokens is not None and args.max_tokens < 1:
        raise SystemExit("max-tokens must be positive")
    if args.timeout is not None and args.timeout <= 0:
        raise SystemExit("timeout must be positive")
    if args.limit is not None and (args.limit < 1 or args.shard_index is None):
        raise SystemExit("--limit must be positive and used with --shard-index")

    try:
        import httpx
    except ImportError as exc:
        raise SystemExit("httpx is required: python -m pip install httpx") from exc

    from indra_belief.model_client import LOCAL_MODELS

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
    print(
        f"[config] model={args.model} served_id={args.model_id} "
        f"max_tokens={args.max_tokens} timeout={args.timeout:g}s "
        f"workers={args.workers} temperature={args.temperature}",
        flush=True,
    )

    shards = select_shards(Path(args.input_dir), args.shard_index)
    prompt = MonolithicPrompt()
    limits = httpx.Limits(
        max_connections=args.workers,
        max_keepalive_connections=args.workers,
    )
    with httpx.Client(limits=limits, timeout=args.timeout) as client:
        for shard in shards:
            code = run_shard(shard, args, client, prompt)
            if code:
                return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
