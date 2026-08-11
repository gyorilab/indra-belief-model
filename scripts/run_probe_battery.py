#!/usr/bin/env python3
"""Run the no-reasoning probe battery with one in-process MLX read per probe.

The MLX stack remains in ``~/.venvs/mlx-serve`` so its optional model runtime
does not alter the project's pinned scientific environment.  All MLX imports
are lazy, which keeps gold loading, prompt rendering, record construction, and
artifact verification importable in ``.venv``.  Each scored prompt disables
thinking, appends A1's forced-prefill suffix, and reads one full-vocabulary
next-token distribution through ``mlx_lm.generate.generate_step``.

A2 is infrastructure only: it stores raw P(token ``correct``), never applies
probe orientation, and emits no metric, threshold, or decision column.  Prefer
``delta_logit`` downstream: in ``data/probe_battery/smoke_fit.jsonl``, 54/64
reads are ``precision_limited`` and 57/64 have one label logprob rounded to
exactly 0.0, so ``p_raw`` saturates while the log-space difference stays exact.
The per-probe field is named ``status`` (not ``lp_status``), as C1 declares.
C1's file-level ``probe_meta`` lives on the first ``_manifest`` sentinel line.
Row subsampling deliberately exposes only seeded ``--sample``/``--seed``
shuffling; the runner has no unshuffled prefix-selection path.

Typical invocations::

    .venv/bin/python scripts/run_probe_battery.py --gold data/benchmark/eval_curation_v1.jsonl --split fit --out /tmp/probes_fit --dry-run
    ~/.venvs/mlx-serve/bin/python scripts/run_probe_battery.py --gold data/benchmark/eval_curation_v1.jsonl --split fit --sample 4 --seed 0 --out data/probe_battery/smoke_fit.jsonl --allow-single-class
    .venv/bin/python scripts/run_probe_battery.py --verify-artifact data/probe_battery/smoke_fit.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# A1 ADAPTER: bind only to the ACTUAL landed battery surface.  A1 owns every
# probe, template, label id, render rule, and the frozen order; this runner adds
# only the consumer-side base annotation required by C1.
from indra_belief.probes.battery import (  # noqa: E402
    LABEL_TOKEN_IDS,
    PROBE_IDS,
    PROBES,
    battery_digest,
    probe_by_id,
    render,
)
from indra_belief.curation import is_gold_correct  # noqa: E402
from indra_belief.logprobs import label_probability  # noqa: E402


DEFAULT_MODEL = "mlx-community/gemma-4-26b-a4b-it-8bit"
BELIEF_BENCHMARK_PATH = ROOT / "data" / "benchmark" / "belief_benchmark.jsonl"
BASE_PROBE_ID = "pol.verdict_direct"

assert BASE_PROBE_ID in PROBE_IDS
assert sum(probe.id == BASE_PROBE_ID for probe in PROBES) == 1

NORMALIZED_ROW_FIELDS = (
    "subject",
    "object",
    "stmt_type",
    "evidence_text",
    "row_index",
    "source_hash",
    "pa_hash",
    "tag",
    "gold_correct",
    "matches_hash",
    "source_api",
)
WIDE_ROW_FIELDS = (
    "row_index",
    "source_hash",
    "pa_hash",
    "subject",
    "object",
    "tag",
    "gold_correct",
    "stmt_type",
    "matches_hash",
    "source_api",
    "elapsed_s",
    "probes",
)
PROBE_RECORD_FIELDS = (
    "p_raw",
    "status",
    "both_observed",
    "precision_limited",
    "log_p_correct",
    "log_p_incorrect",
    "delta_logit",
    "log_label_mass",
    "argmax_token_id",
    "argmax_is_label",
    "secs",
)
_HOLDOUT_OWN_FIELDS = ("subject", "object", "stmt_type", "tag")
_RESUME_PROVENANCE_FIELDS = (
    "source_hash",
    "pa_hash",
    "subject",
    "object",
    "stmt_type",
    "tag",
    "gold_correct",
    "matches_hash",
    "source_api",
)


class ArtifactError(ValueError):
    """A probe artifact violates its declared wide-file contract."""


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else ROOT / resolved


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = _resolve_path(path)
    rows: list[dict[str, Any]] = []
    with resolved.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in {resolved} line {line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL row {line_number} in {resolved} is not an object"
                )
            rows.append(value)
    return rows


def _required_value(row: dict[str, Any], field: str, *, row_index: int) -> Any:
    if field not in row:
        raise ValueError(f"gold row {row_index} is missing required field {field!r}")
    value = row[field]
    if value is None:
        raise ValueError(f"gold row {row_index} has null required field {field!r}")
    return value


def _hash_string(
    value: Any,
    *,
    field: str,
    row_index: int,
    allow_none: bool,
) -> str | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"gold row {row_index} has null required field {field!r}")
    return str(value)


def _normalized_row(
    source: dict[str, Any],
    *,
    row_index: int,
    evidence_text: str,
    pa_hash: Any,
) -> dict[str, Any]:
    """Return exactly A1's four render fields plus join/label payload."""
    if not isinstance(evidence_text, str) or not evidence_text.strip():
        raise ValueError(f"gold row {row_index} has empty evidence_text")
    row = {
        "subject": _required_value(source, "subject", row_index=row_index),
        "object": _required_value(source, "object", row_index=row_index),
        "stmt_type": _required_value(source, "stmt_type", row_index=row_index),
        # Preserve evidence byte-for-byte, including braces.  format_map does not
        # re-scan a substituted value, so no escaping or second formatting pass.
        "evidence_text": evidence_text,
        "row_index": row_index,
        "source_hash": _hash_string(
            source.get("source_hash"),
            field="source_hash",
            row_index=row_index,
            allow_none=False,
        ),
        "pa_hash": _hash_string(
            pa_hash,
            field="pa_hash",
            row_index=row_index,
            allow_none=True,
        ),
        "tag": _required_value(source, "tag", row_index=row_index),
        "gold_correct": bool(is_gold_correct(source.get("tag"))),
        "matches_hash": _hash_string(
            source.get("matches_hash"),
            field="matches_hash",
            row_index=row_index,
            allow_none=True,
        ),
        "source_api": source.get("source_api"),
    }
    assert tuple(row) == NORMALIZED_ROW_FIELDS
    return row


def _normalize_eval_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row_index, source in enumerate(rows):
        evidence = source.get("evidence_text")
        if source.get("pa_hash") is None:
            raise ValueError(f"FIT gold row {row_index} has no pa_hash")
        normalized.append(
            _normalized_row(
                source,
                row_index=row_index,
                evidence_text=evidence,
                pa_hash=source.get("pa_hash"),
            )
        )
    return normalized


def read_eval_curation_v1(path: str | Path) -> list[dict[str, Any]]:
    """Read the FIT gold, taking its own populated ``evidence_text`` directly."""
    rows = _read_jsonl(path)
    if not rows:
        raise ValueError(f"gold file is empty: {_resolve_path(path)}")
    return _normalize_eval_rows(rows)


def join_holdout_evidence(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join TEST evidence on ``str(source_hash)`` and copy no other joined field.

    ``source_hash`` is evidence-grained, not statement-grained.  Benchmark rows
    in one group can disagree on the claim and tag, so the holdout row remains
    authoritative for subject, object, statement type, tag, and gold.  TEST has
    no safely recoverable ``pa_hash``; it is emitted as null and downstream
    clustering uses ``source_hash``.
    """
    benchmark_rows = _read_jsonl(BELIEF_BENCHMARK_PATH)
    evidence_by_hash: dict[str, str] = {}
    for benchmark_index, benchmark_row in enumerate(benchmark_rows):
        source_hash = benchmark_row.get("source_hash")
        evidence = benchmark_row.get("evidence_text")
        if source_hash is None or not isinstance(evidence, str) or not evidence.strip():
            continue
        key = str(source_hash)
        previous = evidence_by_hash.get(key)
        if previous is not None and previous != evidence:
            raise ValueError(
                "belief_benchmark has ambiguous evidence_text for "
                f"source_hash={key!r} (at row {benchmark_index})"
            )
        evidence_by_hash[key] = evidence

    joined: list[dict[str, Any]] = []
    for row_index, holdout_row in enumerate(rows):
        source_hash = _hash_string(
            holdout_row.get("source_hash"),
            field="source_hash",
            row_index=row_index,
            allow_none=False,
        )
        evidence = evidence_by_hash.get(source_hash)
        if evidence is None:
            raise ValueError(
                "belief_benchmark evidence join did not cover holdout row "
                f"{row_index} source_hash={source_hash!r}"
            )
        normalized = _normalized_row(
            holdout_row,
            row_index=row_index,
            evidence_text=evidence,
            pa_hash=None,
        )
        # Load-bearing assertion: ONLY evidence crosses this join.  In
        # particular, adopting the benchmark representative's claim or tag is
        # corrupt because source_hash is not statement-grained.
        for field in _HOLDOUT_OWN_FIELDS:
            assert normalized[field] == holdout_row[field], (
                f"holdout-owned field {field!r} crossed the evidence join"
            )
        joined.append(normalized)
    return joined


def load_gold(path: str | Path) -> tuple[list[dict[str, Any]], str]:
    """Load either split by data shape, returning rows and evidence provenance."""
    raw_rows = _read_jsonl(path)
    if not raw_rows:
        raise ValueError(f"gold file is empty: {_resolve_path(path)}")
    first_evidence = raw_rows[0].get("evidence_text")
    if isinstance(first_evidence, str) and first_evidence.strip():
        return _normalize_eval_rows(raw_rows), "evidence_text"
    return join_holdout_evidence(raw_rows), "belief_benchmark_join"


def select_rows(
    rows: Sequence[dict[str, Any]],
    *,
    sample: int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select rows while preserving their original gold-file ordinals."""
    if sample is not None:
        if sample > len(rows):
            raise ValueError(f"--sample {sample} exceeds the {len(rows)} loaded rows")
        selected = list(rows)
        random.Random(seed).shuffle(selected)
        selected = selected[:sample]
        return selected, {"mode": "sample", "n": len(selected), "seed": seed}
    selected = list(rows)
    return selected, {"mode": "all", "n": len(selected), "seed": None}


def require_two_classes(
    rows: Sequence[dict[str, Any]], *, allow_single_class: bool
) -> None:
    """Refuse a degenerate selection unless the operator marks it as smoke."""
    classes = {row["gold_correct"] for row in rows}
    if len(classes) < 2 and not allow_single_class:
        raise ValueError(
            "single-class selection refused; holdout data are tag-sorted. "
            "Choose a shuffled --sample N --seed S selection, or pass "
            "--allow-single-class for SMOKE ONLY."
        )


def resolve_probes(probe_ids: Sequence[str]) -> tuple[Any, ...]:
    """Resolve a duplicate-free CLI selection through A1's sole registry."""
    if not probe_ids:
        raise ValueError("at least one --probe-id is required")
    if len(set(probe_ids)) != len(probe_ids):
        raise ValueError("--probe-id values must be unique")
    try:
        probes = tuple(probe_by_id(probe_id) for probe_id in probe_ids)
    except KeyError as error:
        raise ValueError(str(error)) from error
    if BASE_PROBE_ID not in probe_ids:
        raise ValueError(
            f"probe selection must include the unique base probe {BASE_PROBE_ID!r}"
        )
    assert sum(probe.id == BASE_PROBE_ID for probe in probes) == 1
    return probes


def _stable_logaddexp(left: float, right: float) -> float:
    """Two-value log-sum-exp using stdlib Python 3.12/3.13.

    The change plan names ``math.logaddexp``, but neither supported interpreter
    exposes that function.  This algebraically equivalent form stays in log
    space and therefore preserves the (-800, -801) underflow case.
    """
    if math.isnan(left) or math.isnan(right):
        return math.nan
    if left == math.inf or right == math.inf:
        return math.inf
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    high = max(left, right)
    return high + math.log1p(math.exp(-abs(left - right)))


def probe_record(
    *,
    probe: Any,
    log_p_correct: float,
    log_p_incorrect: float,
    argmax_token_id: int,
    secs: float | None,
) -> dict[str, Any]:
    """Build one pure per-probe record using the canonical label renormalizer."""
    declared = probe_by_id(probe.id)
    if probe != declared:
        raise ValueError(f"probe {probe.id!r} is not A1's declared registry value")
    log_p_correct = float(log_p_correct)
    log_p_incorrect = float(log_p_incorrect)
    info = label_probability(
        [
            {
                "top": [
                    {"token": "correct", "logprob": log_p_correct},
                    {"token": "incorrect", "logprob": log_p_incorrect},
                ]
            }
        ],
        position=0,
    )
    # With a full-vocabulary read both labels are present and precision is not
    # top-k-limited by construction.  The provenance flags are carried only for
    # shape compatibility; extreme exp underflow may still yield no_label_mass.
    # Prefer delta_logit downstream: the shipped smoke has 54/64
    # precision-limited reads and 57/64 reads with a label logprob rounded to
    # exactly 0.0, while this log-space difference remains exact.
    record = {
        "p_raw": info["p_raw"],
        "status": info["status"],
        "both_observed": info["both_observed"],
        "precision_limited": info["precision_limited"],
        "log_p_correct": log_p_correct,
        "log_p_incorrect": log_p_incorrect,
        "delta_logit": log_p_correct - log_p_incorrect,
        "log_label_mass": _stable_logaddexp(log_p_correct, log_p_incorrect),
        "argmax_token_id": int(argmax_token_id),
        "argmax_is_label": int(argmax_token_id) in LABEL_TOKEN_IDS,
        "secs": None if secs is None else float(secs),
    }
    assert tuple(record) == PROBE_RECORD_FIELDS
    return record


def wide_record(
    *,
    row: dict[str, Any],
    probe_values: dict[str, dict[str, Any]],
    elapsed_s: float | None,
) -> dict[str, Any]:
    """Build C1's one-object-per-gold-row schema."""
    record = {
        "row_index": row["row_index"],
        "source_hash": row["source_hash"],
        "pa_hash": row["pa_hash"],
        # Acceptance checks the holdout claim survived the evidence-only join,
        # so subject and object travel as explicit provenance too.
        "subject": row["subject"],
        "object": row["object"],
        "tag": row["tag"],
        "gold_correct": row["gold_correct"],
        "stmt_type": row["stmt_type"],
        "matches_hash": row["matches_hash"],
        "source_api": row["source_api"],
        "elapsed_s": None if elapsed_s is None else float(elapsed_s),
        "probes": probe_values,
    }
    assert tuple(record) == WIDE_ROW_FIELDS
    return record


def _probe_meta(probes: Sequence[Any]) -> dict[str, dict[str, Any]]:
    meta = {
        probe.id: {
            "family": probe.family,
            "is_base": probe.id == BASE_PROBE_ID,
        }
        for probe in probes
    }
    assert sum(value["is_base"] for value in meta.values()) == 1
    return meta


def build_manifest(
    *,
    gold_path: str | Path,
    split: str,
    model: str,
    rows: Sequence[dict[str, Any]],
    evidence_source: str,
    selection: dict[str, Any],
    probes: Sequence[Any],
    started_at: str | None = None,
) -> dict[str, Any]:
    """Build the first-line manifest shared by dry and scored artifacts."""
    if split not in {"fit", "test"}:
        raise ValueError(f"unknown split: {split!r}")
    probe_ids = [probe.id for probe in probes]
    manifest = {
        "_manifest": True,
        "probe_meta": _probe_meta(probes),
        "gold_path": str(_resolve_path(gold_path).resolve()),
        "split": split,
        "model": model,
        "battery_digest": battery_digest(),
        "n_rows": len(rows),
        "n_dropped": 0,
        "evidence_source": evidence_source,
        "selection": dict(selection),
        "probe_ids": probe_ids,
        "started_at": started_at or datetime.now(timezone.utc).isoformat(),
        "cluster_field": "pa_hash" if split == "fit" else "source_hash",
        "pa_hash_note": (
            "FIT pa_hash is carried from the gold row."
            if split == "fit"
            else "TEST pa_hash is null because holdout_cc has no safely "
            "recoverable pa_hash; cluster TEST on source_hash."
        ),
    }
    return manifest


def _json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _prompt_output_path(out_path: str | Path) -> Path:
    return Path(f"{_resolve_path(out_path)}.prompts.jsonl")


def write_dry_run(
    *,
    out_path: str | Path,
    manifest: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    probes: Sequence[Any],
) -> Path:
    """Render all message pairs without importing a tokenizer or MLX."""
    prompt_path = _prompt_output_path(out_path)
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    dry_manifest = dict(manifest)
    dry_manifest["dry_run"] = True
    n_prompts = 0
    with prompt_path.open("w", encoding="utf-8") as handle:
        handle.write(_json_line(dry_manifest) + "\n")
        for row in rows:
            prompt_map: dict[str, dict[str, str]] = {}
            for probe in probes:
                system, user, prefill_suffix = render(probe, row)
                if "SUBSTRATE HINT" in user:
                    raise ValueError(
                        f"rendered row_index={row['row_index']} contains SUBSTRATE HINT"
                    )
                prompt_map[probe.id] = {
                    "system": system,
                    "user": user,
                    "prefill_suffix": prefill_suffix,
                }
                n_prompts += 1
            prompt_row = wide_record(row=row, probe_values={}, elapsed_s=None)
            prompt_row.pop("probes")
            prompt_row["prompts"] = prompt_map
            handle.write(_json_line(prompt_row) + "\n")
    distinct_pa_hash = len({row["pa_hash"] for row in rows if row["pa_hash"] is not None})
    nonempty_evidence = sum(bool(row["evidence_text"].strip()) for row in rows)
    print(
        f"dry_run_rows={len(rows)} prompts={n_prompts} n_dropped=0 "
        f"nonempty_evidence={nonempty_evidence} "
        f"distinct_pa_hash={distinct_pa_hash} output={prompt_path}"
    )
    return prompt_path


def _load_model(model_id: str):
    """Load the MLX model and tokenizer only in the MLX execution path."""
    from mlx_lm import load

    return load(model_id)


def _assert_template_geometry(tok: Any) -> None:
    """Abort before scoring if A1's recorded tokenizer contract has moved."""
    for word, expected_id in zip(("correct", "incorrect"), LABEL_TOKEN_IDS):
        actual = tok.encode(word, add_special_tokens=False)
        assert actual == [expected_id], (
            f"label token geometry moved: {word!r} encoded as {actual}, "
            f"expected {[expected_id]}"
        )
    rendered = tok.apply_chat_template(
        [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert rendered.endswith("<|channel>thought\n<channel|>"), (
        "disabled-thinking chat template geometry moved"
    )
    assert "<|think|>" not in rendered


def _score_one(
    model: Any,
    tok: Any,
    prompt_text: str,
    label_ids: tuple[int, int],
) -> tuple[float, float, int]:
    """Read two label log-probs and the full-vocab argmax from one MLX step."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    # The rendered chat template already contains its BOS marker.  The live
    # tokenizer currently produces the same ids either way, but spelling this
    # out prevents a future tokenizer default from silently adding a second.
    ids = mx.array(tok.encode(prompt_text, add_special_tokens=False))
    # With no sampler argument, generate_step's default is mx.argmax: the
    # temperature-zero path.  We still derive integrity from logprobs below.
    _yielded_token, logprobs = next(generate_step(ids, model, max_tokens=1))
    # Derive integrity from the distribution itself, independent of any future
    # change to generate_step's default sampler.
    argmax_token_id = int(mx.argmax(logprobs).item())
    return (
        float(logprobs[label_ids[0]].item()),
        float(logprobs[label_ids[1]].item()),
        argmax_token_id,
    )


def _parse_artifact(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    resolved = _resolve_path(path)
    values = _read_jsonl(resolved)
    if not values:
        raise ArtifactError(f"artifact is empty: {resolved}")
    manifest = values[0]
    if manifest.get("_manifest") is not True:
        raise ArtifactError("artifact first line must be the _manifest record")
    rows = values[1:]
    if any(row.get("_manifest") is True for row in rows):
        raise ArtifactError("artifact contains more than one _manifest line")
    return manifest, rows


def _number(value: Any, *, field: str, allow_none: bool) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactError(f"{field} must be numeric" + (" or null" if allow_none else ""))
    return float(value)


def _validate_artifact_parts(
    manifest: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    *,
    require_complete: bool,
) -> dict[str, Any]:
    probe_ids = manifest.get("probe_ids")
    if (
        not isinstance(probe_ids, list)
        or not probe_ids
        or not all(isinstance(value, str) for value in probe_ids)
        or len(set(probe_ids)) != len(probe_ids)
    ):
        raise ArtifactError("manifest probe_ids must be a non-empty unique string list")
    if BASE_PROBE_ID not in probe_ids:
        raise ArtifactError(f"manifest probe_ids omit base probe {BASE_PROBE_ID!r}")
    if manifest.get("battery_digest") != battery_digest():
        raise ArtifactError("manifest battery_digest does not match the landed battery")
    if manifest.get("n_dropped") != 0:
        raise ArtifactError("manifest n_dropped must be 0")
    n_rows = manifest.get("n_rows")
    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows <= 0:
        raise ArtifactError("manifest n_rows must be a positive integer")
    if require_complete and len(rows) != n_rows:
        raise ArtifactError(
            f"artifact row count {len(rows)} does not match manifest n_rows={n_rows}"
        )
    if not require_complete and len(rows) > n_rows:
        raise ArtifactError(
            f"partial artifact has {len(rows)} rows but manifest n_rows={n_rows}"
        )

    selection = manifest.get("selection")
    if not isinstance(selection, dict) or set(selection) != {"mode", "n", "seed"}:
        raise ArtifactError("manifest selection requires exactly mode, n, and seed")
    selection_mode = selection["mode"]
    if selection_mode not in {"all", "sample"}:
        raise ArtifactError("manifest selection mode must be all or sample")
    selection_n = selection["n"]
    if (
        isinstance(selection_n, bool)
        or not isinstance(selection_n, int)
        or selection_n != n_rows
    ):
        raise ArtifactError("manifest selection n must equal manifest n_rows")
    selection_seed = selection["seed"]
    if selection_mode == "sample":
        if isinstance(selection_seed, bool) or not isinstance(selection_seed, int):
            raise ArtifactError("sample selection requires an integer seed")
    elif selection_seed is not None:
        raise ArtifactError(f"{selection_mode} selection requires seed=null")

    meta = manifest.get("probe_meta")
    if not isinstance(meta, dict) or set(meta) != set(probe_ids):
        raise ArtifactError("manifest probe_meta must cover exactly the declared probe_ids")
    base_count = 0
    for probe_id in probe_ids:
        value = meta[probe_id]
        if not isinstance(value, dict):
            raise ArtifactError(f"probe_meta[{probe_id!r}] must be an object")
        if not isinstance(value.get("family"), str) or not isinstance(
            value.get("is_base"), bool
        ):
            raise ArtifactError(
                f"probe_meta[{probe_id!r}] requires family:str and is_base:bool"
            )
        declared = probe_by_id(probe_id)
        if value["family"] != declared.family:
            raise ArtifactError(f"probe_meta family mismatch for {probe_id!r}")
        expected_base = probe_id == BASE_PROBE_ID
        if value["is_base"] != expected_base:
            raise ArtifactError(f"probe_meta is_base mismatch for {probe_id!r}")
        base_count += int(value["is_base"])
    if base_count != 1:
        raise ArtifactError(f"probe_meta must declare exactly one base probe, got {base_count}")

    split = manifest.get("split")
    if split not in {"fit", "test"}:
        raise ArtifactError("manifest split must be 'fit' or 'test'")
    expected_evidence_source = (
        "evidence_text" if split == "fit" else "belief_benchmark_join"
    )
    if manifest.get("evidence_source") != expected_evidence_source:
        raise ArtifactError(
            f"manifest split={split!r} requires evidence_source="
            f"{expected_evidence_source!r}"
        )
    expected_cluster_field = "pa_hash" if split == "fit" else "source_hash"
    if manifest.get("cluster_field") != expected_cluster_field:
        raise ArtifactError(
            f"manifest split={split!r} requires cluster_field="
            f"{expected_cluster_field!r}"
        )
    required_wide = set(WIDE_ROW_FIELDS)
    required_probe = set(PROBE_RECORD_FIELDS)
    seen_row_indices: set[int] = set()
    pa_hashes: set[str] = set()
    status_histogram: Counter[str] = Counter()
    record_timings: list[float] = []
    probe_read_timings: list[float] = []
    missing_secs = 0
    missing_elapsed_s = 0
    argmax_label_count = 0
    n_probe_records = 0

    for ordinal, row in enumerate(rows):
        missing = required_wide - set(row)
        if missing:
            raise ArtifactError(f"artifact row {ordinal} missing fields: {sorted(missing)}")
        if "row_i" in row:
            raise ArtifactError(f"artifact row {ordinal} uses forbidden field row_i")
        row_index = row["row_index"]
        if isinstance(row_index, bool) or not isinstance(row_index, int):
            raise ArtifactError(f"artifact row {ordinal} row_index must be an integer")
        if row_index in seen_row_indices:
            raise ArtifactError(f"duplicated row_index: {row_index}")
        seen_row_indices.add(row_index)
        if not isinstance(row["source_hash"], str) or not row["source_hash"]:
            raise ArtifactError(f"artifact row {row_index} source_hash must be a string")
        pa_hash = row["pa_hash"]
        if split == "fit":
            if not isinstance(pa_hash, str) or not pa_hash:
                raise ArtifactError(f"FIT artifact row {row_index} has null/invalid pa_hash")
            pa_hashes.add(pa_hash)
        elif pa_hash is not None:
            raise ArtifactError(f"TEST artifact row {row_index} must emit pa_hash as null")
        if not isinstance(row["gold_correct"], bool):
            raise ArtifactError(f"artifact row {row_index} gold_correct must be boolean")
        elapsed = _number(
            row["elapsed_s"], field=f"row {row_index} elapsed_s", allow_none=True
        )
        if elapsed is None:
            missing_elapsed_s += 1
        elif not math.isfinite(elapsed) or elapsed < 0.0:
            raise ArtifactError(f"row {row_index} elapsed_s must be finite and non-negative")
        else:
            record_timings.append(elapsed)

        probes = row["probes"]
        if not isinstance(probes, dict) or set(probes) != set(probe_ids):
            raise ArtifactError(
                f"row {row_index} probes must cover exactly declared probe_ids"
            )
        for probe_id in probe_ids:
            probe_value = probes[probe_id]
            if not isinstance(probe_value, dict):
                raise ArtifactError(f"row {row_index} probe {probe_id!r} is not an object")
            missing_probe = required_probe - set(probe_value)
            if missing_probe:
                raise ArtifactError(
                    f"row {row_index} probe {probe_id!r} missing fields: "
                    f"{sorted(missing_probe)}"
                )
            if "lp_status" in probe_value:
                raise ArtifactError(
                    f"row {row_index} probe {probe_id!r} uses forbidden lp_status"
                )
            log_p_correct = _number(
                probe_value["log_p_correct"],
                field=f"row {row_index} probe {probe_id} log_p_correct",
                allow_none=False,
            )
            log_p_incorrect = _number(
                probe_value["log_p_incorrect"],
                field=f"row {row_index} probe {probe_id} log_p_incorrect",
                allow_none=False,
            )
            log_label_mass = _number(
                probe_value["log_label_mass"],
                field=f"row {row_index} probe {probe_id} log_label_mass",
                allow_none=False,
            )
            assert log_p_correct is not None
            assert log_p_incorrect is not None
            assert log_label_mass is not None
            for field, value in (
                ("log_p_correct", log_p_correct),
                ("log_p_incorrect", log_p_incorrect),
                ("log_label_mass", log_label_mass),
            ):
                if not math.isfinite(value):
                    raise ArtifactError(
                        f"row {row_index} probe {probe_id!r} {field} is not finite"
                    )

            delta = _number(
                probe_value["delta_logit"],
                field=f"row {row_index} probe {probe_id} delta_logit",
                allow_none=False,
            )
            assert delta is not None
            if not math.isfinite(delta):
                raise ArtifactError(
                    f"row {row_index} probe {probe_id!r} delta_logit is not finite"
                )
            expected_delta = log_p_correct - log_p_incorrect
            if not math.isclose(delta, expected_delta, rel_tol=1e-12, abs_tol=1e-12):
                raise ArtifactError(
                    f"row {row_index} probe {probe_id!r} delta_logit does not "
                    "match the stored label log-probabilities"
                )
            expected_log_label_mass = _stable_logaddexp(
                log_p_correct, log_p_incorrect
            )
            if not math.isclose(
                log_label_mass,
                expected_log_label_mass,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ArtifactError(
                    f"row {row_index} probe {probe_id!r} log_label_mass does not "
                    "match the stored label log-probabilities"
                )
            status = probe_value["status"]
            if not isinstance(status, str):
                raise ArtifactError(f"row {row_index} probe {probe_id!r} status is not str")
            status_histogram[status] += 1
            if not isinstance(probe_value["both_observed"], bool):
                raise ArtifactError(
                    f"row {row_index} probe {probe_id!r} both_observed is not bool"
                )
            if not isinstance(probe_value["precision_limited"], bool):
                raise ArtifactError(
                    f"row {row_index} probe {probe_id!r} precision_limited is not bool"
                )
            p_raw = probe_value["p_raw"]
            p_value: float | None = None
            if p_raw is not None:
                p_value = _number(
                    p_raw,
                    field=f"row {row_index} probe {probe_id} p_raw",
                    allow_none=False,
                )
                assert p_value is not None
                if not math.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
                    raise ArtifactError(
                        f"row {row_index} probe {probe_id!r} p_raw is outside [0,1]"
                    )
            expected_probability = label_probability(
                [
                    {
                        "top": [
                            {"token": "correct", "logprob": log_p_correct},
                            {"token": "incorrect", "logprob": log_p_incorrect},
                        ]
                    }
                ],
                position=0,
            )
            for field in ("status", "both_observed", "precision_limited"):
                if probe_value[field] != expected_probability[field]:
                    raise ArtifactError(
                        f"row {row_index} probe {probe_id!r} {field} does not "
                        "match label_probability"
                    )
            expected_p_raw = expected_probability["p_raw"]
            if (p_value is None) != (expected_p_raw is None) or (
                p_value is not None
                and expected_p_raw is not None
                and not math.isclose(
                    p_value, expected_p_raw, rel_tol=1e-12, abs_tol=1e-12
                )
            ):
                raise ArtifactError(
                    f"row {row_index} probe {probe_id!r} p_raw does not "
                    "match label_probability"
                )
            argmax_token_id = probe_value["argmax_token_id"]
            if isinstance(argmax_token_id, bool) or not isinstance(argmax_token_id, int):
                raise ArtifactError(
                    f"row {row_index} probe {probe_id!r} argmax_token_id is not int"
                )
            if not isinstance(probe_value["argmax_is_label"], bool):
                raise ArtifactError(
                    f"row {row_index} probe {probe_id!r} argmax_is_label is not bool"
                )
            expected_argmax_is_label = argmax_token_id in LABEL_TOKEN_IDS
            if probe_value["argmax_is_label"] != expected_argmax_is_label:
                raise ArtifactError(
                    f"row {row_index} probe {probe_id!r} argmax_is_label does not "
                    "match argmax_token_id"
                )
            argmax_label_count += int(probe_value["argmax_is_label"])
            n_probe_records += 1
            secs = _number(
                probe_value["secs"],
                field=f"row {row_index} probe {probe_id} secs",
                allow_none=True,
            )
            if secs is None:
                missing_secs += 1
            elif not math.isfinite(secs) or secs < 0.0:
                raise ArtifactError(
                    f"row {row_index} probe {probe_id!r} secs must be finite and non-negative"
                )
            else:
                probe_read_timings.append(secs)

    return {
        "n_records": len(rows),
        "n_distinct_row_index": len(seen_row_indices),
        "n_distinct_pa_hash": len(pa_hashes),
        "n_probe_records": n_probe_records,
        "argmax_is_label_rate": (
            argmax_label_count / n_probe_records if n_probe_records else None
        ),
        "status_histogram": dict(sorted(status_histogram.items())),
        "median_s_per_record": (
            statistics.median(record_timings) if record_timings else None
        ),
        "median_s_per_probe_read": (
            statistics.median(probe_read_timings) if probe_read_timings else None
        ),
        "missing_secs": missing_secs,
        "missing_elapsed_s": missing_elapsed_s,
        "row_indices": seen_row_indices,
    }


def verify_artifact(path: str | Path) -> dict[str, Any]:
    """Validate and summarize one complete wide artifact without importing MLX."""
    manifest, rows = _parse_artifact(path)
    summary = _validate_artifact_parts(manifest, rows, require_complete=True)
    rate = summary["argmax_is_label_rate"]
    record_median = summary["median_s_per_record"]
    probe_read_median = summary["median_s_per_probe_read"]
    print("manifest_lines=1")
    print(f"n_records={summary['n_records']}")
    print(f"n_probe_records={summary['n_probe_records']}")
    print(f"n_distinct_row_index={summary['n_distinct_row_index']}")
    print(f"n_distinct_pa_hash={summary['n_distinct_pa_hash']}")
    print(
        "argmax_is_label_rate="
        + ("null" if rate is None else f"{rate:.6f}")
    )
    print(
        "status_histogram="
        + json.dumps(summary["status_histogram"], sort_keys=True, separators=(",", ":"))
    )
    print(
        "median_s_per_record="
        + ("null" if record_median is None else f"{record_median:.6f}")
    )
    print(
        "median_s_per_probe_read="
        + ("null" if probe_read_median is None else f"{probe_read_median:.6f}")
    )
    print(f"missing_secs={summary['missing_secs']}")
    print(f"missing_elapsed_s={summary['missing_elapsed_s']}")
    return summary


def _resume_rows(
    out_path: Path,
    *,
    expected_manifest: dict[str, Any],
    expected_rows: Sequence[dict[str, Any]],
) -> set[int]:
    manifest, rows = _parse_artifact(out_path)
    for key in (
        "gold_path",
        "split",
        "model",
        "battery_digest",
        "n_rows",
        "n_dropped",
        "evidence_source",
        "selection",
        "probe_ids",
        "probe_meta",
        "cluster_field",
        "pa_hash_note",
    ):
        if manifest.get(key) != expected_manifest.get(key):
            raise ArtifactError(f"--resume manifest mismatch for {key}")
    summary = _validate_artifact_parts(manifest, rows, require_complete=False)
    expected_by_index = {row["row_index"]: row for row in expected_rows}
    if len(expected_by_index) != len(expected_rows):
        raise ArtifactError("selected rows contain duplicated row_index values")
    for existing in rows:
        row_index = existing["row_index"]
        expected = expected_by_index.get(row_index)
        if expected is None:
            raise ArtifactError(
                f"--resume artifact contains unselected row_index={row_index}"
            )
        for field in _RESUME_PROVENANCE_FIELDS:
            if existing.get(field) != expected[field]:
                raise ArtifactError(
                    f"--resume row_index={row_index} provenance mismatch for {field}"
                )
    return set(summary["row_indices"])


def run_scoring(
    *,
    out_path: str | Path,
    manifest: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    probes: Sequence[Any],
    model_id: str,
    resume: bool,
) -> int:
    """Run the lazy MLX half and append complete wide rows incrementally."""
    output = _resolve_path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[int] = set()
    write_manifest = True
    append_needs_newline = False
    mode = "w"
    if resume and output.exists() and output.stat().st_size:
        completed = _resume_rows(
            output,
            expected_manifest=manifest,
            expected_rows=rows,
        )
        write_manifest = False
        mode = "a"
        with output.open("rb") as existing:
            existing.seek(-1, 2)
            append_needs_newline = existing.read(1) != b"\n"
    selected_indices = {row["row_index"] for row in rows}
    if not completed <= selected_indices:
        extra = sorted(completed - selected_indices)
        raise ArtifactError(f"--resume artifact contains unselected row_index values: {extra}")
    pending = [row for row in rows if row["row_index"] not in completed]
    if not pending:
        print(f"resume_complete rows={len(completed)} output={output}")
        return 0

    model, tok = _load_model(model_id)
    _assert_template_geometry(tok)
    with output.open(mode, encoding="utf-8") as handle:
        if write_manifest:
            handle.write(_json_line(manifest) + "\n")
            handle.flush()
        elif append_needs_newline:
            handle.write("\n")
            handle.flush()
        for completed_now, row in enumerate(pending, start=1):
            row_started = time.perf_counter()
            probe_values: dict[str, dict[str, Any]] = {}
            for probe in probes:
                probe_started = time.perf_counter()
                system, user, prefill_suffix = render(probe, row)
                if "SUBSTRATE HINT" in user:
                    raise ValueError(
                        f"rendered row_index={row['row_index']} contains SUBSTRATE HINT"
                    )
                prompt_text = tok.apply_chat_template(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                ) + prefill_suffix
                log_p_correct, log_p_incorrect, argmax_token_id = _score_one(
                    model,
                    tok,
                    prompt_text,
                    LABEL_TOKEN_IDS,
                )
                secs = time.perf_counter() - probe_started
                probe_values[probe.id] = probe_record(
                    probe=probe,
                    log_p_correct=log_p_correct,
                    log_p_incorrect=log_p_incorrect,
                    argmax_token_id=argmax_token_id,
                    secs=secs,
                )
            elapsed_s = time.perf_counter() - row_started
            record = wide_record(
                row=row,
                probe_values=probe_values,
                elapsed_s=elapsed_s,
            )
            handle.write(_json_line(record) + "\n")
            handle.flush()
            if completed_now <= 10 or completed_now % 25 == 0 or completed_now == len(pending):
                print(
                    f"scored={completed_now}/{len(pending)} "
                    f"row_index={row['row_index']} elapsed_s={elapsed_s:.3f}"
                )
    return 0


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the pure CLI, conditionally requiring run inputs outside verify mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--split", choices=("fit", "test"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--probe-id",
        action="append",
        default=None,
        help="repeatable A1 probe id; defaults to all probes in frozen order",
    )
    parser.add_argument("--sample", type=_positive_int, help="shuffle then take N")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-single-class", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-artifact", type=Path)
    args = parser.parse_args(argv)
    if args.probe_id is None:
        args.probe_id = list(PROBE_IDS)

    if args.verify_artifact is not None:
        incompatible = [
            name
            for name, value in (
                ("--gold", args.gold),
                ("--split", args.split),
                ("--out", args.out),
                ("--sample", args.sample),
            )
            if value is not None
        ]
        if args.resume:
            incompatible.append("--resume")
        if args.dry_run:
            incompatible.append("--dry-run")
        if args.allow_single_class:
            incompatible.append("--allow-single-class")
        if incompatible:
            parser.error(
                "--verify-artifact cannot be combined with " + ", ".join(incompatible)
            )
        return args

    missing = [
        name
        for name, value in (("--gold", args.gold), ("--split", args.split), ("--out", args.out))
        if value is None
    ]
    if missing:
        parser.error("the following arguments are required: " + ", ".join(missing))
    if args.resume and args.dry_run:
        parser.error("--resume cannot be combined with --dry-run")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.verify_artifact is not None:
            verify_artifact(args.verify_artifact)
            return 0

        rows, evidence_source = load_gold(args.gold)
        expected_evidence_source = (
            "evidence_text" if args.split == "fit" else "belief_benchmark_join"
        )
        if evidence_source != expected_evidence_source:
            raise ValueError(
                f"--split {args.split} requires evidence_source="
                f"{expected_evidence_source!r}, but {_resolve_path(args.gold)} "
                f"loaded as {evidence_source!r}"
            )
        selected, selection = select_rows(
            rows,
            sample=args.sample,
            seed=args.seed,
        )
        require_two_classes(selected, allow_single_class=args.allow_single_class)
        probes = resolve_probes(args.probe_id)
        manifest = build_manifest(
            gold_path=args.gold,
            split=args.split,
            model=args.model,
            rows=selected,
            evidence_source=evidence_source,
            selection=selection,
            probes=probes,
        )
        if args.dry_run:
            write_dry_run(
                out_path=args.out,
                manifest=manifest,
                rows=selected,
                probes=probes,
            )
            return 0
        return run_scoring(
            out_path=args.out,
            manifest=manifest,
            rows=selected,
            probes=probes,
            model_id=args.model,
            resume=args.resume,
        )
    except (AssertionError, ArtifactError, KeyError, OSError, TypeError, ValueError) as error:
        print(f"run_probe_battery: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
