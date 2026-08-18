#!/usr/bin/env python3
"""Turn scored corpus shards into statement beliefs.

THE MISSING HALF OF THE CORPUS PATH
-----------------------------------
``run_vllm_processed_shards.py`` reads evidences and writes
``{stmt_hash: {source_hash: {verdict, confidence, probe_delta_logit}}}``. Until
this script existed, NOTHING read that shape -- a grep for its fields across the
tree returned writes and nothing else. The corpus path produced verdicts and
stopped; no belief was computed at 60M scale, with or without logits.

This joins the two halves of a shard pair and finishes the job:

    grounded-NNNNNN.jsonl.gz   (input)   source_api, tier, stmt/source hashes
    verdicts-NNNNNN.json.gz    (output)  verdict, confidence, probe_delta_logit
        |
        +-- per-evidence rows --> statement_belief --> {stmt_hash: belief}

WHERE THE LOGITS ENTER
----------------------
A persisted ``probe_delta_logit`` is a RAW log-odds and is meaningless across
serving stacks: the same weights read in-process and over HTTP correlate at
r=0.955 but differ 2.4x in range and disagree in sign on 10% of rows. So it is
converted to an additive weight ONLY through an isotonic registered for THIS
(model, served_model_id), and only then does ``statement_belief`` consume it --
which it already does by default, with no flag, wherever a row carries one.

With no isotonic registered the margins are carried through untouched and every
row keeps its verdict weight. That is the same fail-safe the rest of the
codebase uses, and it is why this script is worth running before any
calibration exists: the beliefs are correct-but-uncalibrated rather than absent.

STREAMING
---------
A statement's evidences never span shards -- the shard writer rolls only at
statement boundaries (build_processed_grounding_shards.py:644) -- so a shard can
be reduced to beliefs and released before the next is opened. That is what makes
60M evidences tractable in constant memory.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _load_runner():
    """Reuse the runner's own job reader and path rules.

    Imported rather than reimplemented: the join is on job identity, and a local
    copy of `iter_jobs`/`output_paths` would be one edit away from joining
    against a shape the runner no longer writes.
    """
    spec = importlib.util.spec_from_file_location(
        "_shard_runner", ROOT / "scripts/run_vllm_processed_shards.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def evidence_rows(job: dict[str, Any], cell: dict[str, Any] | None) -> dict | None:
    """One scored evidence, in the shape statement_belief reads.

    Returns None for an evidence the run never scored, which is NOT the same as
    one scored as wrong: an unscored evidence must not be credited in either
    direction, so it is dropped and counted rather than defaulted.
    """
    if not cell:
        return None
    verdict = cell.get("verdict")
    if verdict not in {"correct", "incorrect"}:
        return None

    # Tier comes from the INPUT job, because the published cell does not carry
    # it and statement_belief's gate turns on it: `no_text` rows are excluded
    # entirely, and the two deterministic tiers are credited differently from an
    # LLM read. A tier1 job carries its own tier; anything the model actually
    # read is an LLM tier by construction.
    tier1 = job.get("tier1_result") or {}
    tier = tier1.get("tier") if tier1 else None
    if not tier:
        tier = "llm"

    return {
        "source_api": job.get("source_api") or None,
        "verdict": verdict,
        "tier": tier,
        "evidence_hash": str(job.get("source_hash")),
        "probe_delta_logit": cell.get("probe_delta_logit"),
    }


def apply_weights(rows: list[dict], calibration, stats: dict) -> list[dict]:
    """Attach a calibrated additive weight wherever a margin was measured.

    No calibration -> the rows are returned untouched and every one keeps its
    verdict weight. A margin is never mapped through a curve fitted on another
    stack; that produced saturated 0/1 output when it was tried.
    """
    if calibration is None:
        return rows
    from indra_belief.probes.calibration import calibrate_probe
    from indra_belief.probes.reader import ProbeReading

    out = []
    for row in rows:
        margin = row.get("probe_delta_logit")
        if isinstance(margin, (int, float)) and not isinstance(margin, bool):
            try:
                reading = calibrate_probe(
                    ProbeReading(p_raw=float("nan"), delta_logit=float(margin)),
                    record_id=row.get("evidence_hash") or "row",
                    calibration=calibration,
                )
                row = {**row, "weight_of_evidence": reading.weight_of_evidence}
                stats["n_weighted"] += 1
            except Exception:
                stats["n_weight_failed"] += 1
        out.append(row)
    return out


def beliefs_for_shard(runner, input_path: Path, results_path: Path, *,
                      soft, calibration, priors, stats: dict) -> dict[str, float]:
    """Reduce one shard pair to {stmt_hash: belief}."""
    from indra_belief.statement_belief import statement_belief

    with gzip.open(results_path, "rt", encoding="utf-8") as fh:
        verdicts = json.load(fh)

    by_stmt: dict[str, list[dict]] = {}
    for job in runner.iter_jobs(input_path):
        stmt_hash = str(job.get("stmt_hash"))
        source_hash = str(job.get("source_hash"))
        cell = (verdicts.get(stmt_hash) or {}).get(source_hash)
        row = evidence_rows(job, cell)
        if row is None:
            stats["n_unscored"] += 1
            continue
        # A scored row carrying no margin is the signature of a tokenizer whose
        # label token does not match what the reader scans for (a leading space
        # or a trailing quote both yield None, silently). On a new serving stack
        # that can be EVERY row while verdicts land perfectly.
        if row.get("probe_delta_logit") is None:
            stats["n_null_margin"] += 1
        by_stmt.setdefault(stmt_hash, []).append(row)

    out: dict[str, float] = {}
    for stmt_hash, rows in by_stmt.items():
        rows = apply_weights(rows, calibration, stats)
        result = statement_belief(rows, priors, soft=soft)
        out[stmt_hash] = float(result.belief)
        stats["weighting"][result.weighting] = (
            stats["weighting"].get(result.weighting, 0) + 1
        )
    stats["n_statements"] += len(out)
    stats["n_evidence"] += sum(len(v) for v in by_stmt.values())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True,
                    help="prepared grounding shards (grounded-*.jsonl.gz)")
    ap.add_argument("--results-dir", required=True,
                    help="scored shards from run_vllm_processed_shards.py")
    ap.add_argument("--model", default="vllm-local")
    ap.add_argument("--variant", default=None,
                    help="scoring variant the run used; defaults to the runner's")
    ap.add_argument("--served-model-id", default=None,
                    help="override the served id used to resolve the isotonic")
    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="the --limit the SCORING run used; output filenames carry "
                         "it (verdicts-NNNNNN.limit-K.json.gz), so omitting it here "
                         "makes every lookup miss and yields an empty table")
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-calibrated", action="store_true",
                    help="refuse to write an uncalibrated belief table")
    args = ap.parse_args()

    runner = _load_runner()
    variant_name = args.variant or runner.DEFAULT_VARIANT

    from indra_belief.calibration_constants import calibration_banner, calibration_for
    from indra_belief.noise_model import RECALIBRATED_PRIORS
    from indra_belief.probes.calibration import (
        _calibration_at, sentence_calibration_path_for,
    )
    from indra_belief.scorers.monolithic import scorer as mono

    prompt_sha256 = hashlib.sha256(
        mono.VARIANTS[variant_name].system_prompt.encode("utf-8")
    ).hexdigest()

    # 1. the BELIEF profile: (model, prompt sha) -> the two verdict log-LRs.
    #    Without it every belief is hard-gate; the run is still valid but far
    #    less trustworthy (ECE 0.237 against 0.045) and nothing downstream can
    #    tell the difference, so the banner is printed unconditionally.
    soft = calibration_for(args.model, prompt_sha256=prompt_sha256)
    calibrated, banner = calibration_banner(args.model, prompt_sha256)
    print(banner, flush=True)

    # 2. the IN-CALL isotonic: (model, served id) -> margin becomes a weight.
    #    A DIFFERENT artifact from the profile above, fitted on a different
    #    quantity, and either can exist without the other.
    class _Probe:
        model_name = args.model
        backend = "openai_compat"
        _guard = None
        config = {"model_id": args.served_model_id or "", "max_top_logprobs": 1024}

    path = sentence_calibration_path_for(_Probe()) if args.served_model_id else None
    calibration = _calibration_at(path) if path else None
    print(
        f"[weights] in-call isotonic: "
        f"{'registered — margins become additive weights' if calibration else 'NONE — every row keeps its verdict weight'}",
        flush=True,
    )
    if args.require_calibrated and not (calibrated and calibration):
        raise SystemExit(
            "refusing to write: --require-calibrated needs BOTH a fitted belief "
            "profile and a registered in-call isotonic for this stack"
        )

    input_dir, results_dir = Path(args.input_dir), Path(args.results_dir)
    shards = sorted(input_dir.glob("grounded-*.jsonl.gz"))
    if args.shard_index is not None:
        shards = [p for p in shards
                  if int(runner.SHARD_RE.search(p.name).group(1)) == args.shard_index]
    if not shards:
        raise SystemExit(f"no input shards found in {input_dir}")

    stats = {"n_statements": 0, "n_evidence": 0, "n_unscored": 0, "n_weighted": 0,
             "n_weight_failed": 0, "n_shards": 0, "n_missing_results": 0,
             "n_null_margin": 0, "weighting": {}}
    table: dict[str, float] = {}
    for shard in shards:
        index = int(runner.SHARD_RE.search(shard.name).group(1))
        results_path, _ = runner.output_paths(results_dir, index, args.limit)
        if not results_path.exists():
            stats["n_missing_results"] += 1
            continue
        shard_table = beliefs_for_shard(
            runner, shard, results_path, soft=soft, calibration=calibration,
            priors=RECALIBRATED_PRIORS, stats=stats,
        )
        # The streaming join rests on "a statement never spans shards" -- true of
        # today's writer, which commits the open shard before writing an
        # oversized statement's jobs. It is a CROSS-FILE invariant this script
        # cannot enforce and dict.update() would violate in silence, replacing a
        # whole-statement belief with one computed from a fraction of its
        # evidence. Checked, not assumed.
        collided = set(shard_table) & set(table)
        if collided:
            raise SystemExit(
                f"{len(collided)} stmt_hash values appear in more than one shard "
                f"(first: {sorted(collided)[0]}). Merging would silently discard "
                "the earlier shard's evidence for those statements."
            )
        table.update(shard_table)
        stats["n_shards"] += 1

    # A run that found no shard results at all is a configuration error wearing
    # a successful exit code -- most often a --limit the scoring run used and
    # this one was not told about. An empty table is never a legitimate answer.
    if stats["n_shards"] == 0:
        raise SystemExit(
            f"no shard results were read ({stats['n_missing_results']} input "
            f"shards had no matching output in {results_dir}). If the scoring run "
            "used --limit, pass the same --limit here; the filenames carry it."
        )

    # An isotonic that was registered but never actually applied produces a table
    # indistinguishable from a calibrated one, with a manifest naming the curve.
    if calibration is not None and stats["n_weighted"] == 0:
        raise SystemExit(
            f"an isotonic is registered but NOT ONE row was weighted "
            f"({stats['n_weight_failed']} failed, {stats['n_null_margin']} carried "
            "no margin). Every belief here is verdict-weighted; publishing it "
            "under a calibrated manifest would misdescribe the whole table."
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, sort_keys=True) + "\n")
    manifest = {
        # The number is meaningless without the configuration that produced it,
        # and this is the only place a consumer can read that.
        "model": args.model,
        "variant": variant_name,
        "prompt_sha256": prompt_sha256,
        "belief_profile_fitted": bool(soft),
        "in_call_isotonic": path.name if path else None,
        "served_model_id": args.served_model_id,
        "key": "stmt_hash — the first column of the corpus TSV, NOT re-derived here",
        **stats,
    }
    out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"\n  statements={stats['n_statements']:,} evidence={stats['n_evidence']:,} "
        f"weighted={stats['n_weighted']:,} unscored={stats['n_unscored']:,}\n"
        f"  weighting={stats['weighting']}\n  wrote {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
