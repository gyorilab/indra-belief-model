"""Run prepared grounding shards through vLLM server or offline inference.

The script keeps three pieces of recovery machinery:

* each failed job whose failure a rerun could change is retried before being
  finalized with ``verdict="error"``;
* every scored job is appended to a run-keyed ``.partial.jsonl`` and skipped
  after a restart, so an interruption costs the last few rows rather than the
  whole shard;
* the final gzip JSON dictionary is written to a unique temporary file, fsynced
  and atomically renamed, so a crash cannot publish a half-written result and a
  file that does not read back is rescored rather than skipped.

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
import hashlib
import json
import math
import os
import pickle
import re
import sys
import tempfile
import time
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, NamedTuple


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from indra_belief.vllm_offline import OfflineVllmClient

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
UNPARSEABLE_REPLY = "unparseable model response"
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
                        limit: int | None = None,
                        *, allow_limited: bool = False) -> Path | None:
    """Find the scored file for a shard by its limit-qualified name.

    Output names carry the limit (`verdicts-000000.limit-400.json.gz`), so a
    reader that reconstructs the unlimited name from a remembered --limit
    misses EVERY shard and exits cleanly over an empty result rather than
    raising an error.

    The name is fully determined by (index, limit), so the DEFAULT is the exact
    match and nothing else. A non-exact glob fallback can silently join a
    1,000-job smoke test in place of a 50,000-job shard: the remaining 49,000
    evidences fall to `n_unscored` and every statement past the limit vanishes
    from the belief table under a manifest that looks healthy.

    ``allow_limited`` is for the one caller that genuinely does not know the
    scoring run's --limit -- the calibration fit, which reads whatever rows it
    can find -- so the leniency is stated at the call site that wants it rather
    than applied to every consumer.
    """
    exact = output_path(output_dir, shard_index, limit)
    if exact.exists():
        return exact
    if not allow_limited:
        return None
    candidates = sorted(output_dir.glob(f"verdicts-{shard_index:06d}*.json.gz"))
    return candidates[0] if len(candidates) == 1 else None


def shard_tag(shard_index: int, limit: int | None) -> str:
    tag = f"{shard_index:06d}"
    if limit is not None:
        tag += f".limit-{limit}"
    return tag


def output_path(
    output_dir: Path,
    shard_index: int,
    limit: int | None,
) -> Path:
    return output_dir / f"verdicts-{shard_tag(shard_index, limit)}.json.gz"


def meta_path_for(results_path: Path) -> Path:
    """The provenance sidecar that travels beside a scored shard.

    A sidecar rather than a field inside the payload or a longer filename: the
    published dictionary's shape and the `verdicts-NNNNNN[.limit-K].json.gz`
    name are both consumed by readers that must keep working unchanged.
    """
    name = results_path.name
    if name.endswith(".json.gz"):
        name = name[: -len(".json.gz")]
    return results_path.with_name(f"{name}.meta.json")


def publish_atomically(path: Path, write_body) -> None:
    """Write `path` so a reader never sees a partial or unwritten one.

    Every file this runner publishes -- the shard AND its sidecar -- requires
    both properties:

    * DURABILITY. The rename is metadata and the body is data, so a node crash
      can commit the rename while the extents are still dirty -- leaving a
      zero-length or NUL-filled file that the completion check reads as a
      finished shard forever. fsync the file before the rename and the directory
      after it; two fsyncs against a shard that took minutes to score.
    * A UNIQUE staging name. A name derived only from the target -- `{final}.tmp`
      -- is shared by every process writing that target. Two writers can open
      the SAME inode with O_TRUNC and interleave their writes; the loser's rename
      then raises FileNotFoundError on a shard that was fully scored.

    NOT SWEPT, deliberately. SIGKILL between mkstemp and rename leaks one
    `.{name}.XXXX.tmp`: harmless to every reader (no reader globs staging names)
    but cumulative over preemptions. Sweeping this shard's stale staging files
    on publish would delete the in-flight staging file of a concurrent writer of
    the same shard -- the exact collision the unique name exists to prevent. The
    `test_two_writers_of_one_sidecar_do_not_destroy_each_others_staging_file`
    test pins that collision; telling a live one from an abandoned one needs an
    age cutoff with no measurement behind it. Orphans are for the operator to
    remove between runs, when nothing is writing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, staged = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                      dir=path.parent)
    temporary = Path(staged)
    try:
        with os.fdopen(handle, "wb") as raw:
            write_body(raw)
            raw.flush()
            os.fsync(raw.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def write_shard_meta(path: Path, meta: dict[str, Any]) -> None:
    """Publish the provenance sidecar durably and atomically.

    Same staging discipline as the shard itself, for the same two reasons: the
    sidecar name is derived only from the shard, so a plain `{name}.tmp` is
    shared by every process writing it, and an unfsynced sidecar can survive a
    crash as zero bytes -- which is not a configuration and not an absent one
    either, so `read_shard_provenance` withholds that shard from every later
    run until a human looks at it.

    ORDERING IS THE CALLER'S. `run_shard` writes this before the shard, so an
    interruption between the two leaves a sidecar with no shard -- ignored,
    because the shard is rescored anyway -- rather than a shard whose
    configuration is unknown.
    """
    def body(raw):
        raw.write(
            (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )

    publish_atomically(path, body)


def write_final_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Publish a complete gzip JSON dictionary with one atomic rename."""
    def body(raw):
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6) as compressed:
            text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            compressed.write((text + "\n").encode("utf-8"))

    publish_atomically(path, body)


class ShardWithheld(Exception):
    """A fault about ONE shard's files, carrying the reason a human needs.

    Raised where the fault is seen and turned into `ShardOutcome(2, reason)` by
    `run_shard`, because that is the answer this runner already gives every
    other per-shard problem: the shard is withheld and named at the end, and the
    other ~1,199 keep their run. The two callers below reach it from opposite
    directions -- a published shard nobody can read, a sidecar nobody can read
    -- and neither may be answered with a default, since one default rescores
    and overwrites a good shard and the other lets a fault pass as agreement.
    """


def _gzip_frame_is_plausible(path: Path) -> bool:
    """Reject the two zero-cost wreckage shapes without reading the body.

    A resume checks every already-published shard before the first new job; the
    full decompress below is ~9MB of shared scratch each. An empty file and a
    file whose tail extents are lost to a crash (NULs, so a CRC32/ISIZE trailer
    of all zeros) are both decidable from two seeks, and those are the shapes an
    unfsynced writer can leave behind.

    An OSError PROPAGATES. This says "the bytes on disk are wreckage", and a
    file that cannot be opened right now -- ESTALE, EIO, EPERM during a resume
    walk over ~1,200 shards on shared scratch -- has said nothing about its
    bytes. Answering False there lets a transient mount fault rescore and
    overwrite a good shard.
    """
    size = path.stat().st_size
    if size < 20:  # 10-byte header + 8-byte trailer + at least one block
        return False
    with path.open("rb") as fh:
        if fh.read(2) != b"\x1f\x8b":
            return False
        fh.seek(-8, os.SEEK_END)
        return fh.read(8) != b"\x00" * 8


def published_output_is_readable(path: Path) -> bool:
    """Whether an existing output file is a finished shard or wreckage.

    `final_path.exists()` cannot tell the two apart: a truncated or NUL-filled
    gzip left by a crash reads as complete, so
    the shard is never regenerated and the corruption surfaces one stage later
    as a BadGzipFile in the belief build.

    Decompressed, not parsed: reading the stream to EOF verifies gzip's own
    CRC32 and length trailer without materializing a 50,000-cell dictionary. The
    cheap frame check runs first and rejects both zero-length shapes -- a 0-byte
    file on `size < 20`, a gzip of empty content on its all-zero trailer -- but
    does not replace this read: every shard already on disk was written by the
    pre-fsync `gzip.open` + `replace` writer, where a kill mid-write leaves a
    VALID PREFIX with a garbage tail whose 8 trailer bytes are not zeros, and
    only reading the stream catches that.

    The final `}` is the last of the three: a member that decompresses cleanly
    to something which is not a finished object -- whitespace, a payload
    serialized short -- passes both checks above and is not a shard. Tracked
    across chunks rather than within the last one -- a payload of exactly 1 (mod
    1MiB) bytes ends its last read on the lone trailing newline, which rstrips
    to empty and condemns a perfectly good shard to be rescored.

    Only a CORRUPT-BYTES failure answers False. `_gzip_frame_is_plausible`
    defines why a bare OSError says nothing about the shard's bytes. Such a fault
    raises `ShardWithheld`, because the caller's response to False is to
    regenerate ~50,000 verdicts and overwrite the published file, and that
    replacement is not the same data (batched vLLM at temperature 0 is not
    bitwise reproducible on an MoE, and `_cell_priority` resolves repeats
    differently from the writer that produced the files on disk). It withholds
    THIS shard rather than aborting the run, matching the sidecar-conflict branch
    for the same kind of per-shard problem.
    """
    last = b""
    try:
        if not _gzip_frame_is_plausible(path):
            return False
        with gzip.open(path, "rb") as fh:
            while chunk := fh.read(1 << 20):
                stripped = chunk.rstrip()
                if stripped:
                    last = stripped[-1:]
    except (gzip.BadGzipFile, EOFError, zlib.error):
        return False
    except OSError as exc:
        raise ShardWithheld(
            f"could not read the published shard: {path}: {exc}"
        ) from exc
    return last == b"}"


class MonolithicPrompt:
    """Reuse the repo's commit-first disconfirm_relnature workflow."""

    # indra_belief.verdict.parse_verdict is the SINGLE reader for the live path,
    # the batch replay, and every profile. A hand-rolled parser here would make
    # this runner a fourth reading of the corpus and would differ under
    # truncation.
    #
    # prepared_execution.relation_mismatch_note OWNS the label table and the
    # mismatch sentence, so a byte change there must move the live and batch
    # prompts together. A local copy in this file is therefore FORBIDDEN.
    #
    # The class's own docstring says "reuse the repo's workflow": delegate to
    # those owners rather than importing their parts.
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

        # A variant is a COHERENT SET — verdict-first prompt, thinking suppressed,
        # temperature 0. Borrowing one member of that set and hard-coding the rest
        # creates an inconsistent request; reading the whole set from one object
        # keeps the three from drifting and preserves the no-CoT path.
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
        model that may put the answer in either channel; the READING is
        CENTRALISED, so a reply that scores here scores identically on the live
        scorer and the batch replay.

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
        # The tier1 verdict is DATA on the input job; rereading it produces the
        # same malformed dict every time.
        return {**base, "verdict": None, "confidence": None, "error": "bad tier1",
                "retryable": False}

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
            # A job with no user_message raises the same way three more times.
            "retryable": False,
        }

    variant = prompt.variant
    # The variant's own declarations, not this runner's defaults. The variant
    # declares reasoning because gemma-4's served chat template defaults thinking
    # ON, and at 60M evidences an unasked-for CoT is the entire bill.
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
            "error": UNPARSEABLE_REPLY,
            # RETRYABLE. A continuously-batched vLLM server is not bitwise
            # reproducible at temperature 0 -- the floating-point reduction
            # order follows the batch composition, and gemma-4-26B-A4B is an MoE
            # whose expert routing varies with batch shape -- so an identical
            # reissue lands in a different batch and genuinely can come back
            # parseable. Whether the reissues DID come back different is what
            # `score_job_with_retries` classifies on; this flag only buys them.
            "retryable": True,
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
        retryable = _is_retryable(exc)
    return {
        **base,
        "verdict": None,
        "confidence": None,
        "error": error,
        "retryable": retryable,
    }


def _is_retryable(exc: BaseException) -> bool:
    """Whether reissuing the IDENTICAL request could produce a different result.

    Only a client error says no: a 4xx is the server rejecting the request
    BYTES, and those do not change between attempts, so an over-length prompt is
    over-length three more times at the full request cost. 408 and 429 are the
    two 4xx codes that DO change with time. Everything else -- a transport
    failure, a timeout, a 5xx -- is worth another attempt.

    RECOGNISED ONLY FROM AN HTTP STATUS, so this is a --backend server rule. An
    offline engine raises a bare exception with no `.response`, and its rejection
    of an over-length prompt is a ValueError indistinguishable from a transient
    one, so every offline failure is retried. The cost is bounded: `_issue` has
    already narrowed a failed batch to the single conversation that caused it,
    so the retries fall on that row rather than its --workers neighbours.
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int) and 400 <= status < 500 and status not in (408, 429):
        return False
    return True



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
    # max 15 over 40 records on MLX) and widens on demand. This path widens
    # silently, so a stack whose losing-label rank differs costs one extra call
    # per affected record with no counter behind it.
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
        ceiling = PROBE_TOP_LOGPROBS
        if ceiling <= top_logprobs:
            return None
        try:
            return _issue(ceiling)
        except Exception:
            return None
    except Exception:
        return None


def _reply_signature(row: dict[str, Any]) -> tuple | None:
    """What an unreadable reply was, or None for any other failure.

    `response_preview` is the model's message object truncated to 4,000
    characters as a diagnostic cap, so two attempts with the same signature
    agree on their first 4,000 characters and on their finish reason -- not on
    the whole reply. That prefix is the evidence the deterministic class rests
    on, and it is worth what it is: a degenerate repetition that diverges only
    past character 4,001 is classed as reproduced.

    A transport failure has no reply and answers None, which keeps a MIXED
    sequence -- one timeout, three refusals -- out of the deterministic class.
    """
    if row.get("error") != UNPARSEABLE_REPLY:
        return None
    return (row.get("response_preview"), row.get("finish_reason"))


def _reply_is_reproduced(replies: list[tuple | None]) -> bool:
    """Whether every attempt this row was given returned the identical reply.

    That is the only deterministic-failure evidence available at 60M scale: a
    reply the parser cannot read is deterministic for given prompt bytes
    whatever the batch does, and the reissues are what distinguish it from a
    batch-composition artifact. With --retries 0 one attempt is the whole
    experiment; passing it is the operator asserting a reissue is not worth its
    generation, so the single sample is taken at face value.
    """
    return bool(replies) and None not in replies and len(set(replies)) == 1


def score_job_with_retries(
    job: dict[str, Any],
    *,
    retries: int,
    **score_kwargs: Any,
) -> dict[str, Any]:
    """Score one job, retrying only the failures a rerun could change.

    A CLIENT ERROR is terminal on the first attempt: a 4xx, a job with no
    user_message, a malformed tier1 verdict. Those are properties of the request
    or the input row, so retrying them cost `retries` more full generations
    apiece and could not change the answer -- at 60M jobs a 1% rate is 1.8M
    pointless requests, each up to the registry's 8192-token ceiling.

    An unparseable REPLY is not in that set: batched vLLM at temperature 0 is
    not bitwise reproducible, so the same bytes reissued into a different batch
    can parse, and it keeps its retries.

    IT IS STILL CLASSIFIED ON WHAT CAME BACK. The reissues are the experiment:
    if every attempt returned the SAME unreadable reply, that reply is a
    property of the prompt bytes -- a refusal, a degenerate repetition -- and no
    further rerun changes it. Replies that DIFFER across attempts stay transient,
    which is the case the retry argument is actually about.

    The retained `attempts` is what the model was actually asked, not what the
    loop planned, so a row that stopped at one is distinguishable from a row
    that exhausted four transport failures. `error_class` records WHICH of the
    two kinds of failure exhausted. The publication gate counts both -- a shard
    of nothing but refusals is the corpus-poisoning case -- and the class is
    what lets its refusal name the right remedy: rerun the command, or decide
    that failures no rerun can clear are worth publishing.
    """
    allowed = retries + 1 if job.get("needs_llm", True) else 1
    row: dict[str, Any] | None = None
    attempt = 0
    replies: list[tuple | None] = []
    for attempt in range(1, allowed + 1):
        row = score_job(job, **score_kwargs)
        if valid_result(row):
            return row
        replies.append(_reply_signature(row))
        if not row.get("retryable", True):
            break
    assert row is not None
    row["attempts"] = attempt
    row["error_class"] = (
        "deterministic"
        if not row.get("retryable", True) or _reply_is_reproduced(replies)
        else "transient"
    )
    return row


def _cell_priority(cell: dict[str, Any]) -> tuple:
    """Choose between two cells for one (stmt_hash, source_hash).

    `statement_belief._dedup_priority`'s rule, on the published cell shape: a
    semantic read beats an error, and between two reads the conservative
    any-incorrect-wins choice is taken -- the same rule the belief combiner
    applies to a repeated evidence, so the two cannot disagree. File order never
    decides, which is exactly what last-writer-wins made it do.
    """
    verdict = cell.get("verdict")
    return (
        0 if verdict in VALID_VERDICTS else 1,
        0 if verdict == "incorrect" else 1 if verdict == "correct" else 2,
        0 if cell.get("probe_delta_logit") is not None else 1,
    )


def finalize(
    input_path: Path,
    latest: dict[str, dict[str, Any]],
    limit: int | None,
    stmt_hashes: set[int] | None = None,
    stats: dict[str, int] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Build the hash dictionary, retaining exhausted failures as errors.

    The run is keyed on job_id and the published file is keyed on
    (stmt_hash, source_hash), and those are NOT the same identity: under
    `--all-evidence` a statement can carry the same evidence twice, which is two
    jobs and one cell. Writing the second over the first discarded a scored
    verdict and its margin by file order -- an errored duplicate could erase a
    good read, and the shard-end count was the collapsed one, so nothing showed
    it. A repeat is reconciled and counted in ``stats["duplicate_pairs"]``.
    """
    payload: dict[str, dict[str, dict[str, Any]]] = {}
    # WHICH KIND of failure survived for each retained pair, kept BESIDE the
    # payload rather than inside the cell: the publication gate reports the
    # split so its refusal can name the remedy for each class, and the on-disk
    # shard format is consumed by readers that must keep working unchanged --
    # the ~1,200 shards the live run published have no such field. Keyed on the
    # pair, and rewritten when a duplicate wins, so the class always describes
    # the cell that was published.
    error_classes: dict[tuple[str, str], str] = {}
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
        by_source = payload.setdefault(stmt_hash, {})
        existing = by_source.get(source_hash)
        if existing is not None:
            if stats is not None:
                stats["duplicate_pairs"] = stats.get("duplicate_pairs", 0) + 1
            chosen = min((existing, cell), key=_cell_priority)
        else:
            chosen = cell
        if chosen is cell:
            if cell.get("verdict") == "error":
                # Default "transient": an unrecognised failure is assumed
                # re-runnable, the same assumption `row.get("retryable", True)`
                # makes upstream.
                error_classes[(stmt_hash, source_hash)] = str(
                    result.get("error_class") or "transient"
                )
            else:
                error_classes.pop((stmt_hash, source_hash), None)
        by_source[source_hash] = chosen
    if stats is not None:
        stats["errors_deterministic"] = sum(
            1 for value in error_classes.values() if value == "deterministic"
        )
        stats["errors_transient"] = (
            len(error_classes) - stats["errors_deterministic"]
        )
    return payload


def stmt_hash_filter_digest(stmt_hashes: set[int] | None) -> str | None:
    """A short, order-free name for the --gene-stmt-hashes filter, or None."""
    if stmt_hashes is None:
        return None
    digest = hashlib.sha256()
    for value in sorted(stmt_hashes):
        digest.update(str(value).encode("ascii"))
        digest.update(b",")
    return f"{len(stmt_hashes)}:{digest.hexdigest()[:16]}"


def run_provenance(args, prompt: MonolithicPrompt,
                   stmt_hashes: set[int] | None) -> dict[str, Any]:
    """What a scored shard cannot say about itself.

    The published file is a bare {stmt_hash: {source_hash: cell}} dict, so the
    model, the prompt and the acquisition route of `probe_delta_logit` are
    nowhere in it, so stage 4 (`scripts/build_corpus_beliefs.py`) treats
    --model/--variant as unverifiable assertions: a run scored under
    `disconfirm_relnature_rf` and believed under the default resolves a profile
    fitted for a prompt it never sent, then publishes that claim in its manifest.
    The filter belongs here too -- the output NAME
    carries the --limit but not `--gene-stmt-hashes`, so a filtered file
    otherwise satisfies an unfiltered rerun and every remaining shard is skipped
    as complete.

    The GENERATION CEILING belongs here too, because `partial_path` digests this
    dict: without them, interrupting a shard and restarting with a different
    --max-tokens loaded the first ceiling's rows as done and published one file
    whose verdicts came from two ceilings, with nothing able to see it. The
    digest already blocked a variant switch for exactly that reason.
    `temperature` is the EFFECTIVE temperature of the VERDICT call -- what
    score_job sends -- since the variant overrides the flag. The [Complex]
    relation-nature sub-call is not covered: it fixes its own temperature and
    this field does not record it.
    """
    variant = prompt.variant
    return {
        "model": args.model,
        "served_model_id": args.model_id,
        "variant": variant.name,
        "max_tokens": args.max_tokens,
        "temperature": (
            variant.temperature if variant.temperature is not None
            else args.temperature
        ),
        "prompt_sha256": hashlib.sha256(
            prompt.system_prompt.encode("utf-8")
        ).hexdigest(),
        "margin_route": (
            "pol.verdict_incall" if variant.in_call_label_logprobs
            else "pol.verdict_direct" if getattr(args, "probe", False) else None
        ),
        "limit": args.limit,
        "stmt_hash_filter": stmt_hash_filter_digest(stmt_hashes),
    }


PROVENANCE_KEYS = (
    "model", "served_model_id", "variant", "prompt_sha256", "stmt_hash_filter",
)


def read_shard_provenance(meta_path: Path) -> dict[str, Any] | None:
    """What a published shard records about itself, or None if it records nothing.

    ABSENCE AND FAILURE ARE DIFFERENT ANSWERS, and only one of them is consent.
    A bare `except Exception: return None` makes a transient mount fault --
    ESTALE/EIO/EPERM during a resume walk over ~1,200 shards on shared scratch,
    or a sidecar whose extents are lost mid-write -- indistinguishable from
    "this shard predates sidecars". That routes the fault onto the sidecar-less
    path, where absence never disagrees with anything, and accepts the shard as
    matching a configuration nobody read.

    FileNotFoundError alone is the unrecorded case: the live published shards
    have no sidecar at all, and they must stay joinable. Every other failure
    withholds the shard it belongs to.
    """
    try:
        with meta_path.open(encoding="utf-8") as fh:
            recorded = json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise ShardWithheld(
            f"the provenance sidecar could not be read: {meta_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(recorded, dict):
        raise ShardWithheld(
            f"the provenance sidecar is not a JSON object: {meta_path}: "
            f"got {type(recorded).__name__}"
        )
    return recorded


def provenance_conflict(meta_path: Path,
                        provenance: dict[str, Any]) -> str | None:
    """The first recorded field this run disagrees with, if any."""
    recorded = read_shard_provenance(meta_path)
    if recorded is None:
        return None
    for key in PROVENANCE_KEYS:
        if key in recorded and recorded[key] != provenance.get(key):
            return f"{key}={recorded[key]!r} on disk, {provenance.get(key)!r} now"
    return None


def partial_path(output_dir: Path, shard_index: int, limit: int | None,
                 provenance: dict[str, Any]) -> Path:
    """The per-job append log for one shard of one run.

    Keyed on the RUN, not just the shard: a name carrying only the index and the
    limit lets a `disconfirm_relnature_rf` shard restart under the default, load
    the first variant's rows as done, and publish one file whose verdicts come
    from two prompts, with margins from a variant that reads none. The same
    argument is why `run_provenance` carries max_tokens and temperature: a
    restart under a different ceiling is the same defect with a quieter symptom.
    """
    digest = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    return output_dir / f".verdicts-{shard_tag(shard_index, limit)}.{digest}.partial.jsonl"


def load_partial(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    """Keep the latest attempt per job and drop incomplete rows.

    Any line that is not a complete row is dropped. A crash-truncated tail is
    the expected cause, but corruption can occur anywhere in the append log.
    """
    latest: dict[str, dict[str, Any]] = {}
    dropped = 0
    if not path.exists():
        return latest, dropped
    with path.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
                latest[str(row["job_id"])] = row
            except (json.JSONDecodeError, KeyError, TypeError):
                dropped += 1
                continue
    return latest, dropped


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


class ShardOutcome(NamedTuple):
    """What one shard did, and -- if it was withheld -- why.

    The reason travels with the code because `main` prints one summary at the
    end: a run over 1,200 shards that withholds three of them has to name those
    three, and by then their console output is far out of scroll.
    """

    code: int
    reason: str | None = None


def run_shard(
    input_path: Path,
    args,
    client,
    prompt: MonolithicPrompt,
    stmt_hashes: set[int] | None = None,
    provenance: dict[str, Any] | None = None,
) -> ShardOutcome:
    from tqdm import tqdm

    match = SHARD_RE.search(input_path.name)
    assert match
    shard_index = int(match.group(1))
    output_dir = Path(args.output_dir)
    final_path = output_path(output_dir, shard_index, args.limit)
    meta_path = meta_path_for(final_path)
    if provenance is None:
        provenance = run_provenance(args, prompt, stmt_hashes)

    if final_path.exists():
        try:
            conflict = provenance_conflict(meta_path, provenance)
            complete = not conflict and published_output_is_readable(final_path)
        except ShardWithheld as exc:
            # A FILE THIS RUN COULD NOT READ SAYS NOTHING ABOUT ITS CONTENTS, so
            # neither "complete" nor "wreckage" is available and rescoring would
            # overwrite a shard whose bytes were never in question. Withheld and
            # named, like every other per-shard problem here.
            reason = f"shard {shard_index}: {exc}"
            print(f"shard {shard_index} NOT rescored: {exc}")
            return ShardOutcome(2, reason)
        if conflict:
            # WITHHELD, NOT FATAL, by the same argument as the error gate: one
            # shard whose sidecar disagrees must not cost the other 1,199 their
            # run. Refusing this shard is still mandatory -- the output name
            # cannot express the difference, so continuing would read as
            # complete over rows this configuration never scored.
            reason = (
                f"shard {shard_index} was scored under a different "
                f"configuration ({conflict}); point --output-dir somewhere else"
            )
            print(f"shard {shard_index} NOT rescored: {reason}: {final_path}")
            return ShardOutcome(2, reason)
        if complete:
            print(f"skip completed shard {shard_index}: {final_path}")
            return ShardOutcome(0)
        print(
            f"rescoring shard {shard_index}: {final_path} exists but does not "
            "read back — a crash between the rename and writeback leaves a file "
            "that only LOOKS complete"
        )

    total = sum(1 for _ in iter_jobs(input_path, args.limit, stmt_hashes))
    output_dir.mkdir(parents=True, exist_ok=True)
    partial = partial_path(output_dir, shard_index, args.limit, provenance)
    latest, dropped_lines = load_partial(partial)
    done = {key for key, row in latest.items() if valid_result(row)}
    ensure_append_boundary(partial)

    pending = (
        job for job in iter_jobs(input_path, args.limit, stmt_hashes)
        if job_id(job) not in done
    )
    started = time.perf_counter()
    errors = 0
    llm = 0
    tier1 = 0
    endpoint = args.base_url.rstrip("/") + "/chat/completions"

    print(
        f"shard={shard_index} jobs={total:,} resumed={len(done):,} "
        f"workers={args.workers} retries={args.retries}"
    )
    if dropped_lines:
        print(
            f"partial={partial.name} dropped_lines={dropped_lines:,} "
            "(not complete rows)"
        )
    progress = tqdm(total=total, initial=len(done), desc="Scoring", unit="job")

    # BLOCK-BUFFERED, deliberately. The rejected per-row-flush alternative is a
    # line-buffered append, not "an fsync per job"; its per-row overhead is
    # single-digit microseconds. No figure is quoted because two machines here
    # disagree by ~18x on the same comparison -- it is a property of the disk,
    # not of the code, and the decision does not turn on it.
    #
    # The cost of having no log at all is the whole shard: an interruption
    # discarded up to 50,000 scored rows, and a scheduler slot shorter than one
    # shard makes zero net progress while every restart looks healthy. With the
    # block-buffered partial log, a crash costs the last ~8KB of rows instead of
    # all of them.
    with partial.open("a") as partial_fh:
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
                    partial_fh.write(json.dumps(row) + "\n")
                    latest[str(row["job_id"])] = row
                    errors += int(not valid_result(row))
                    llm += int(row.get("source") == "llm")
                    tier1 += int(row.get("source") == "tier1")
                    progress.update(1)
                    progress.set_postfix(llm=llm, tier1=tier1, errors=errors)
                    submit_one()

    progress.close()
    elapsed = time.perf_counter() - started
    counts: dict[str, int] = {}
    payload = finalize(input_path, latest, args.limit, stmt_hashes, stats=counts)
    retained_errors = sum(
        1
        for by_source in payload.values()
        for cell in by_source.values()
        if cell.get("verdict") == "error"
    )
    evidence_results = sum(len(by_source) for by_source in payload.values())
    duplicates = counts.get("duplicate_pairs", 0)
    transient_errors = counts.get("errors_transient", 0)
    deterministic_errors = counts.get("errors_deterministic", 0)

    # A SHARD THAT MOSTLY FAILED IS NOT AN ANSWER. Every job in flight during a
    # vLLM restart exhausts its attempts in milliseconds and finalizes as
    # verdict="error"; publishing that seals it, because the next run skips a
    # completed shard and there is no path that rescores only the failed cells.
    # Downstream those evidences are dropped as unscored and the statement is
    # published with a belief computed from a fraction of its reads -- a wrong
    # number rather than an absent one. Below the threshold one bad row still
    # must not block a shard, which is why this is a fraction and not zero.
    #
    # BOTH CLASSES COUNT. Gating on the transient class alone breaks the gate's
    # purpose: a client returning the same unreadable reply to every job produces
    # a shard whose every cell is verdict="error" at exit 0, so a systematically
    # broken prompt or input enters the corpus silently. The unclearable-failure
    # wedge is broken at the other end: the only escape is the operator raising
    # --max-error-fraction, which makes publication a deliberate act rather than
    # a default.
    #
    # The SPLIT is still reported, because the two remedies differ: a transient
    # failure clears on a rerun of the same command, a deterministic one -- a
    # client error, a row with no user_message, a reply that came back IDENTICAL
    # on every attempt -- never will.
    exhausted_errors = transient_errors + deterministic_errors
    # DENOMINATOR IS THE RETAINED PAIR COUNT, not the job count. The numerator
    # comes from finalize's error_classes, which is keyed on
    # (stmt_hash, source_hash) after duplicate pairs collapse; `total` counts
    # JOBS. Under --all-evidence one statement can carry the same evidence
    # twice, so the two bases differ by `duplicates` and the fraction would be
    # computed across a boundary. `evidence_results` is the same basis as the
    # numerator: one entry per retained cell.
    graded = evidence_results
    if graded and exhausted_errors / graded > args.max_error_fraction:
        rate = exhausted_errors / graded
        # Rounded UP. Printing the truncated rate is worse than printing none:
        # a 3-pair shard with one failure has rate 0.3333..., and an operator
        # who follows a printed "0.3333" sets a threshold the rate still
        # exceeds, so the rerun refuses again with the same advice.
        clearing = math.ceil(rate * 10_000) / 10_000
        reason = (
            f"{exhausted_errors:,}/{graded:,} evidence pairs "
            f"({rate:.1%}) exhausted their attempts "
            f"(TRANSIENT {transient_errors:,}, "
            f"DETERMINISTIC {deterministic_errors:,}), above "
            f"--max-error-fraction {args.max_error_fraction:.1%}"
        )
        print(
            f"shard {shard_index} NOT published: {reason}. The scored rows are "
            f"kept in {partial.name}, so a rerun re-pays for the "
            f"{exhausted_errors:,} exhausted rows and NOT for the "
            f"{graded - exhausted_errors:,} that succeeded -- resume skips a "
            "row only when its result was valid, so both classes are rescored. "
            f"That is worth doing for the {transient_errors:,} TRANSIENT ones, "
            f"which can come back different. The {deterministic_errors:,} "
            "DETERMINISTIC ones cannot: no rerun changes a client error, a row "
            "with no user_message, or a reply that came back identical on every "
            "attempt. Publishing those has to be a deliberate act -- rerun with "
            f"--max-error-fraction {clearing:.4f} or higher to bake them into "
            "the corpus knowingly."
        )
        return ShardOutcome(2, f"shard {shard_index}: {reason}")

    write_shard_meta(
        meta_path,
        {
            **provenance,
            "n_jobs": total,
            "n_errors": retained_errors,
            "n_errors_transient": transient_errors,
            "n_errors_deterministic": deterministic_errors,
            "n_duplicate_pairs": duplicates,
            "n_cells": evidence_results,
            "n_partial_lines_dropped": dropped_lines,
        },
    )
    write_final_atomic(final_path, payload)
    partial.unlink(missing_ok=True)
    print(
        f"completed shard {shard_index}: {evidence_results:,} evidence results "
        f"for {len(payload):,} statements from {total:,} jobs in "
        f"{elapsed / 60:.2f} minutes "
        f"({evidence_results / max(elapsed, 1e-9):.2f} jobs/s), "
        f"errors={retained_errors:,} (transient={transient_errors:,} "
        f"deterministic={deterministic_errors:,}) duplicate_pairs={duplicates:,}"
    )
    print(f"output={final_path}")
    return ShardOutcome(0)


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
        help="Additional attempts after a RETRYABLE failed LLM result "
             "(default: 3). A client error — a 4xx, a job with no user_message "
             "— is terminal on the first attempt.",
    )
    parser.add_argument(
        "--max-error-fraction",
        type=float,
        default=0.01,
        help="Refuse to publish a shard whose exhausted-failure rate exceeds "
             "this (default: 0.01), counting BOTH the transient failures and "
             "the deterministic ones. A published shard is skipped forever, so "
             "a window of server trouble is sealed in as missing evidence and "
             "every affected statement published with a depressed belief, while "
             "a systematically broken prompt would otherwise bake a shard of "
             "verdict=\"error\" into the corpus at exit 0. Raising this is the "
             "one way to publish failures no rerun can clear, and is meant to "
             "be a deliberate act: the refusal prints the split and the rate to "
             "clear.",
    )
    # max-tokens and timeout DEFAULT TO THE REGISTRY, matching
    # scripts/run_rasmachine_monolithic.py. Caller-local 1000/180 defaults
    # silently override the registry entry for every run: a 1000-token cap
    # truncates 16.7% of calls under the production
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


def offline_max_logprobs(model_config: dict[str, Any], variant,
                         probe: bool) -> int | None:
    """The widest logprob window this engine must accept.

    vLLM's own default is 20, so an undeclared engine rejects every request that
    carries a window -- after the model is loaded. Three terms, three reasons:

    * the registry's `max_top_logprobs` (1024) is an UPPER BOUND on what any
      caller of this entry may ask for -- the direct probe's losing label was
      measured at rank 42/83/168 -- not a mirror of the live server's flag.
      Nothing in the tree records how that server was started, and the runbook's
      corpus recipe starts it at `--max-logprobs 128`, the window this run
      actually sends. Declaring the bound in-process only guarantees the offline
      engine is never the binding constraint; it does not make the two backends
      accept the same set of requests, since a 128-capped server rejects bodies
      this engine would take.
    * the variant's in-call window is on EVERY request the run sends.
    * the probe ceiling only under --probe, because that is the only path that
      widens on demand; with the flag off nothing reaches it.
    """
    from indra_belief.probes.reader import PROBE_TOP_LOGPROBS

    widths = [
        int(model_config.get("max_top_logprobs") or 0),
        int(variant.in_call_label_logprobs or 0),
        PROBE_TOP_LOGPROBS if probe else 0,
    ]
    return max(widths) or None


def preflight(client, endpoint: str, model_id: str, prompt, *,
              probe: bool = False) -> None:
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

    synthetic_job = {
        "subject": "A",
        "object": "B",
        "stmt_type": "Activation",
        "evidence_text": "A activates B.",
        "user_message": "CLAIM: A [Activation] B\nEVIDENCE: A activates B.",
    }
    direct_probe = bool(probe and not variant.in_call_label_logprobs)
    if direct_probe:
        from indra_belief.probes.reader import (
            PROBE_FIRST_TRY_TOP_LOGPROBS,
            build_probe_request,
        )

        body = build_probe_request(
            synthetic_job,
            model_id=model_id,
            top_logprobs=PROBE_FIRST_TRY_TOP_LOGPROBS,
            inline_extra_body=True,
        )
    else:
        body = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": "Reply with the single word ok."}
            ],
            "max_tokens": 4,
            "temperature": (
                variant.temperature if variant.temperature is not None else 0.0
            ),
            **reasoning_wire_keys(variant.reasoning_effort),
        }
        if variant.in_call_label_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = variant.in_call_label_logprobs

    try:
        response = client.post(endpoint, json=body, timeout=120)
    except Exception as exc:
        # NAME THE TRANSPORT THAT FAILED. The offline backend contacts no HTTP
        # endpoint, so blaming --base-url for an in-process engine's refusal
        # sent the operator to debug a server that was never involved.
        transport = getattr(client, "transport_description", None)
        probe_hint = (
            f"\n  the direct probe needs --max-logprobs "
            f"{PROBE_FIRST_TRY_TOP_LOGPROBS}; restart with that capacity, or "
            "drop --probe to run without this measurement."
            if direct_probe else ""
        )
        raise SystemExit(
            f"[preflight] cannot reach {transport or endpoint}: "
            f"{type(exc).__name__}: {exc}\n"
            + ("  the in-process engine refused the first request; --base-url "
               "is not used on this backend."
               if transport else
               "  is the server up, and does --base-url point at it?")
            + probe_hint
        ) from None

    if response.status_code >= 400:
        detail = response.text[:400]
        hint = ""
        if direct_probe:
            hint = (
                f"\n  the direct probe needs --max-logprobs "
                f"{PROBE_FIRST_TRY_TOP_LOGPROBS}; restart with that capacity, "
                "or drop --probe to run without this measurement."
            )
        elif variant.in_call_label_logprobs and "logprob" in detail.lower():
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

    if direct_probe:
        from indra_belief.probes.reader import (
            PROBE_TOP_LOGPROBS,
            ProbeReadError,
            ProbeTopKError,
            probe_reading_from_payload,
        )

        # This proves the FIRST-TRY width only. A missing label is the normal
        # widen-on-demand case: the run retries at PROBE_TOP_LOGPROBS (1024),
        # which a server launched at --max-logprobs 128 would still reject. The
        # preflight deliberately does not spend a second request to discover it.
        try:
            reading = probe_reading_from_payload(
                response.json(), top_k=PROBE_FIRST_TRY_TOP_LOGPROBS
            )
        except ProbeTopKError as exc:
            print(
                f"[preflight] direct probe route works, but the first-try "
                f"top-{PROBE_FIRST_TRY_TOP_LOGPROBS} window missed a label "
                f"({exc}); the run will widen on demand to "
                f"top-{PROBE_TOP_LOGPROBS}",
                flush=True,
            )
        except ProbeReadError as exc:
            raise SystemExit(
                f"[preflight] the direct probe response was unreadable: "
                f"{type(exc).__name__}: {exc}\n"
                f"  restart with --max-logprobs "
                f"{PROBE_FIRST_TRY_TOP_LOGPROBS}, or drop --probe to run "
                "without this measurement."
            ) from None
        except Exception as exc:
            raise SystemExit(
                f"[preflight] the direct probe response could not be decoded: "
                f"{type(exc).__name__}: {exc}\n"
                f"  restart with --max-logprobs "
                f"{PROBE_FIRST_TRY_TOP_LOGPROBS}, or drop --probe to run "
                "without this measurement."
            ) from None
        else:
            print(
                f"[preflight] direct probe readable on this stack "
                f"(delta_logit {reading.delta_logit:+.3f})",
                flush=True,
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
        # The preflight must issue a REAL SCORING REQUEST and parse it with the
        # runner's own reader, `label_margin_from_payload`. A 'logprobs came back'
        # check is NOT enough: the reader locates the margin by the EMITTED TOKEN
        # matching a label, so a tokenizer that emits " correct" with a leading
        # space, or "correct\"" with the quote attached, returns None -- silently,
        # per row, forever. Verdicts still land, the run looks healthy, and 60M
        # margins are null. MEASURED: both variants return None today. A
        # `build_probe_request` is NOT a substitute: the probe route forces the
        # label to generated position 0 with a prefill and looks it up among that
        # position's alternatives, so it CANNOT exhibit the emitted-token
        # mismatch. The in-call route scans for the first position whose EMITTED
        # token is a label, which is exactly what a leading space or an attached
        # quote breaks.
        try:
            system, messages = prompt.request(synthetic_job)
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
    if not 0 <= args.max_error_fraction <= 1:
        raise SystemExit("--max-error-fraction must be in [0, 1]")

    from indra_belief.model_client import LOCAL_MODELS, canonical_model_name

    # CANONICALISE FIRST. The registry resolves aliases, so a raw lookup can
    # reject `vllm-local` -> `vllm-gemma-4-26b` with "unknown model registry
    # entry" even though the alias resolves everywhere else. The calibration
    # profile is keyed on the CANONICAL name too, so resolving here makes an
    # aliased run and a canonical run the same run.
    args.model = canonical_model_name(args.model)
    if args.model not in LOCAL_MODELS:
        raise SystemExit(f"unknown model registry entry: {args.model}")
    model_config = LOCAL_MODELS[args.model]
    if model_config.get("backend") == "vllm_offline":
        raise SystemExit(
            f"model registry entry {args.model!r} declares backend "
            f"{model_config['backend']!r}, which this shard runner does not "
            "select through --model; use 'vllm-gemma-4-26b' instead and keep "
            "--backend offline for the in-process engine"
        )
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
    # those lines report properties OF this object, so construction must precede
    # them.
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
    # Hashed off the CONSTRUCTED PROMPT OBJECT, not a separate import of one
    # variant's constant. Such a constant can disagree with --variant and report
    # the calibration status of a prompt this run never uses — a banner that is
    # confidently wrong is worse than no banner.
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
    provenance = run_provenance(args, prompt, stmt_hashes)
    # This runner intentionally posts raw OpenAI bodies through a transport
    # instead of calling ModelClient.call: its retry, partial-append, and
    # per-row failure pipeline is per job. The shared wire contract is already
    # recorded by indra_belief.probes.reader.build_probe_request and
    # indra_belief.model_client.reasoning_wire_keys.
    if args.backend == "offline":
        client_context = OfflineVllmClient(
            args.model_id,
            batch_size=args.workers,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_logprobs=offline_max_logprobs(model_config, prompt.variant,
                                              bool(args.probe)),
            timeout=args.timeout,
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

    # A WITHHELD SHARD MUST NOT HALT THE OTHER 1,199. Returning on the first
    # non-zero code makes one bad window cost every shard queued behind it --
    # far more expensive than the missing-evidence defect the gate protects
    # against, and invisible until someone reads the exit code. Each withheld
    # shard is named at the end and the run exits non-zero ONCE.
    withheld: list[str] = []
    with client_context as client:
        preflight(
            client,
            args.base_url.rstrip("/") + "/chat/completions",
            args.model_id,
            prompt,
            probe=bool(args.probe),
        )
        for shard in shards:
            outcome = run_shard(shard, args, client, prompt, stmt_hashes,
                                provenance)
            if outcome.code:
                withheld.append(outcome.reason or str(shard))
    if withheld:
        print(f"\n{len(withheld)} of {len(shards)} shards were NOT published:",
              flush=True)
        for reason in withheld:
            print(f"  {reason}", flush=True)
        print("  a shard withheld for exhausted failures keeps its scored rows "
              "in the .partial.jsonl log beside the output, so rerunning the "
              "same command re-pays for every exhausted row and none of the "
              "successful ones -- resume skips a row only when its result was "
              "valid. TRANSIENT rows can come back different; DETERMINISTIC "
              "ones cannot, and clear only by raising --max-error-fraction to "
              "the rate each refusal prints, which publishes them knowingly. "
              "A shard withheld for a disagreeing sidecar was never scored by "
              "this run and needs a different --output-dir. A shard withheld "
              "because its published output or sidecar could not be READ was "
              "left untouched on purpose -- it is a storage fault to "
              "investigate, not a shard to rescore over.",
              flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
