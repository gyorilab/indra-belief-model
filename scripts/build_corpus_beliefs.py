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
serving stacks: over 1,075 paired reads the same weights in-process and over
HTTP correlate at r=0.935, differ 2.2x in range, and DISAGREE IN SIGN on 20.6%
of rows (`data/probe_battery/http_base1_scores.json`). So it is
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
statement boundaries (``build_processed_grounding_shards.py::commit_current_shard``) -- so a shard can
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
sys.path.insert(0, str(ROOT / "scripts"))


def _load_runner():
    """Reuse the runner's own job reader and path rules.

    Imported rather than reimplemented: the join is on job identity, and a local
    copy of `iter_jobs`/`resolve_results_path` would be one edit away from joining
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
            except Exception as exc:
                stats["n_weight_failed"] += 1
                # The REASON, once. A bare counter hid a total failure behind a
                # number: an in-call artifact loaded fine and then raised "X
                # column order does not match probe_ids" on every single row,
                # and the only symptom was n_weight_failed climbing. One
                # recorded reason turns "nothing weighted" from a mystery into
                # a diagnosis.
                stats.setdefault("first_weight_error",
                                 f"{type(exc).__name__}: {exc}")
        out.append(row)
    return out


def check_margin_route(calibration, margin_route: str, variant_name: str,
                       path) -> None:
    """Refuse an isotonic filed under the OTHER route's registry.

    `_calibration_at` accepts either route's artifact by design, so the route can
    only be checked where it is known -- here. The two are not interchangeable:
    probe knots span -1.70..+1.61 while in-call margins run ~3x wider (median
    |13.22|), so reading an in-call margin through the probe curve returns
    0.0000/1.0000 for every row -- weighted, counted, never an error.

    Lived below the print it protects, inside main(), where no test could reach
    it: neutering it to `if False:` left the whole bridge suite green.
    """
    if calibration is None:
        return
    if tuple(calibration.probe_ids) != (margin_route,):
        raise SystemExit(
            f"{path.name} was fitted on {calibration.probe_ids!r} but variant "
            f"{variant_name!r} produces {margin_route!r} margins. Applying it "
            "would saturate every reading to 0 or 1 without erroring."
        )


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
            # SPLIT, because the two have different causes and different
            # remedies. A missing cell is a join problem -- the wrong results
            # directory, the wrong generation. An "error" cell is a scored shard
            # that FAILED on this evidence, and every one of them lowers its
            # statement's belief by removing a read, so a run whose error count
            # is not tiny is publishing depressed numbers rather than absent
            # ones. Only the split makes that visible in the manifest.
            if cell and cell.get("verdict") == "error":
                stats["n_error_cell"] += 1
            else:
                stats["n_missing_cell"] += 1
            continue
        # A scored row carrying no margin is the signature of a tokenizer whose
        # label token does not match what the reader scans for (a leading space
        # or a trailing quote both yield None, silently). On a new serving stack
        # that can be EVERY row while verdicts land perfectly.
        if row.get("probe_delta_logit") is None:
            stats["n_null_margin"] += 1
        by_stmt.setdefault(stmt_hash, []).append(row)

    # The PROJECTION is export_belief_table's, not a second copy of it. That
    # function already encodes the two rules this table must not get wrong --
    # an unscored statement is omitted rather than defaulted (Statement.from_json
    # defaults a missing belief to 1.0, so a placeholder publishes "certainly
    # true" for something never read), and a hash carrying two different beliefs
    # is REPORTED rather than resolved by last-writer-wins. Applied per shard so
    # the streaming property survives.
    from export_belief_table import build_table

    records = []
    for stmt_hash, rows in by_stmt.items():
        rows = apply_weights(rows, calibration, stats)
        result = statement_belief(rows, priors, soft=soft)
        records.append({"indra_matches_hash": stmt_hash, "belief": result.belief})
        stats["weighting"][result.weighting] = (
            stats["weighting"].get(result.weighting, 0) + 1
        )
    out, diagnostics = build_table(records)
    for key in ("n_unscored_omitted", "n_without_matches_hash",
                "n_hash_collisions_with_differing_belief"):
        stats[key] = stats.get(key, 0) + diagnostics[key]
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
    ap.add_argument("--model", default="vllm-gemma-4-26b")
    ap.add_argument("--variant", default=None,
                    help="scoring variant the run used; defaults to the runner's")
    ap.add_argument("--served-model-id", default=None,
                    help="override the served id used to resolve the isotonic")
    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="the --limit the SCORING run used; output filenames carry "
                         "it (verdicts-NNNNNN.limit-K.json.gz) and the join is "
                         "against that exact generation, so omitting it here "
                         "misses every shard and the run refuses")
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-calibrated", action="store_true",
                    help="refuse to write an uncalibrated belief table")
    ap.add_argument("--stmt-hash-filter", default=None,
                    help="the stmt_hash_filter digest the SCORING run recorded, "
                         "if it was filtered (--gene-stmt-hashes). Omitted "
                         "asserts an unfiltered run. CHECKABLE ONLY AGAINST A "
                         "SIDECAR: a results directory whose shards recorded one "
                         "is refused until this names the same digest, because "
                         "every evidence outside the filter would otherwise land "
                         "in n_missing_cell under a manifest that never says the "
                         "table covers a subset. Shards scored before the runner "
                         "wrote sidecars cannot be checked at all; they join with "
                         "the assertion unverified and are counted in the "
                         "manifest's n_shards_without_provenance")
    ap.add_argument("--min-shard-coverage", type=float, default=0.99,
                    help="refuse to write a table that joined fewer than this "
                         "fraction of the input shards (default: 0.99). INDRA's "
                         "Statement.from_json defaults a missing belief to 1.0, "
                         "so a table covering 1%% of the corpus does not read as "
                         "1%% covered downstream -- it reads as 99%% certainly "
                         "true. Lower it deliberately to build over a partial "
                         "scoring run")
    args = ap.parse_args()

    # CANONICALISE THE SAME WAY THE RUNNER DOES. `run_vllm_processed_shards.main`
    # canonicalises before writing the sidecar, so scoring and believing through
    # the same live alias -- `--model vllm-local`, kept deliberately in
    # `model_client._MODEL_ALIASES` -- recorded "vllm-gemma-4-26b" and asserted
    # "vllm-local", and a correct join exited with the provenance refusal below.
    from indra_belief.model_client import canonical_model_name

    args.model = canonical_model_name(args.model)

    runner = _load_runner()
    variant_name = args.variant or runner.DEFAULT_VARIANT

    from indra_belief.calibration_constants import calibration_banner, calibration_for
    from indra_belief.noise_model import RECALIBRATED_PRIORS
    from indra_belief.probes.calibration import (
        _calibration_at, incall_calibration_path_for, sentence_calibration_path_for,
    )
    from indra_belief.probes.reader import DIRECT_PROBE_ID, IN_CALL_PROBE_ID
    from indra_belief.scorers.monolithic import scorer as mono

    variant = mono.VARIANTS[variant_name]
    prompt_sha256 = hashlib.sha256(
        variant.system_prompt.encode("utf-8")
    ).hexdigest()

    # 1. the BELIEF profile: (model, prompt sha) -> the two verdict log-LRs.
    #    Without it every belief is hard-gate; the run is still valid but far
    #    less trustworthy (ECE 0.237 against 0.045) and nothing downstream can
    #    tell the difference, so the banner is printed unconditionally.
    soft = calibration_for(args.model, prompt_sha256=prompt_sha256)
    calibrated, banner = calibration_banner(args.model, prompt_sha256)
    print(banner, flush=True)

    # 2. the isotonic for the ROUTE the margins were acquired on:
    #    (model, served id) -> margin becomes a weight. A DIFFERENT artifact
    #    from the profile above, fitted on a different quantity, and either can
    #    exist without the other.
    #
    #    THE ROUTE DECIDES WHICH REGISTRY. `probe_delta_logit` holds an IN-CALL
    #    margin when the variant reads the label from the scoring response, and
    #    a separate-PROBE margin when it does not and the run passed --probe.
    #    The two registries are not interchangeable: probe knots span
    #    -1.70..+1.61 while in-call margins run ~3x wider (median |13.22|), so
    #    reading an in-call margin through the probe curve returns 0.0000/1.0000
    #    for every row -- weighted, counted, never an error. The live path
    #    (`replace_sentence_score`) refuses that swap by name; this one used to
    #    commit it.
    class _Probe:
        model_name = args.model
        backend = "openai_compat"
        _guard = None
        config = {"model_id": args.served_model_id or "", "max_top_logprobs": 1024}

    if variant.in_call_label_logprobs:
        margin_route, path_for = IN_CALL_PROBE_ID, incall_calibration_path_for
    else:
        margin_route, path_for = DIRECT_PROBE_ID, sentence_calibration_path_for
    path = path_for(_Probe()) if args.served_model_id else None
    calibration = _calibration_at(path) if path else None
    check_margin_route(calibration, margin_route, variant_name, path)
    # NAMED BY ROUTE, not by one of the two routes. This line and the manifest
    # key both said "in-call" while `margin_route` can be the separate-probe
    # curve -- reachable through the documented `--variant
    # disconfirm_relnature_rf` -- so the artifact contradicted itself about
    # which quantity it was fitted on, one layer above the mis-filed-curve
    # confusion check_margin_route exists to refuse.
    print(
        f"[weights] {margin_route} isotonic: "
        f"{'registered — margins become additive weights' if calibration else 'NONE — every row keeps its verdict weight'}",
        flush=True,
    )
    if args.require_calibrated and not (calibrated and calibration):
        raise SystemExit(
            "refusing to write: --require-calibrated needs BOTH a fitted belief "
            f"profile and a registered {margin_route} isotonic for this stack"
        )

    input_dir, results_dir = Path(args.input_dir), Path(args.results_dir)
    shards = sorted(input_dir.glob("grounded-*.jsonl.gz"))
    if args.shard_index is not None:
        shards = [p for p in shards
                  if int(runner.SHARD_RE.search(p.name).group(1)) == args.shard_index]
    if not shards:
        raise SystemExit(f"no input shards found in {input_dir}")

    stats = {"n_statements": 0, "n_evidence": 0, "n_unscored": 0,
             "n_error_cell": 0, "n_missing_cell": 0, "n_weighted": 0,
             "n_weight_failed": 0, "n_shards": 0, "n_missing_results": 0,
             "n_null_margin": 0, "weighting": {}}
    table: dict[str, float] = {}
    asserted = {
        "model": args.model,
        "variant": variant_name,
        "prompt_sha256": prompt_sha256,
        "served_model_id": args.served_model_id,
        # The output NAME carries --limit but cannot carry --gene-stmt-hashes, so
        # a filtered results directory joins against an unfiltered build with
        # every non-gene evidence falling to n_missing_cell -- under a manifest
        # that says nothing about a subset. The runner records the digest; None
        # here asserts the run was unfiltered, which is the common case and the
        # one that must not pass by accident.
        "stmt_hash_filter": args.stmt_hash_filter,
    }
    unrecorded = 0
    for shard in shards:
        index = int(runner.SHARD_RE.search(shard.name).group(1))
        results_path = runner.resolve_results_path(results_dir, index, args.limit)
        if results_path is None:
            stats["n_missing_results"] += 1
            continue
        # --model/--variant are ASSERTIONS about a run that already happened,
        # and getting one wrong resolves a belief profile fitted for a prompt
        # the scoring run never sent -- then publishes that claim in the
        # manifest. The scoring run records what it actually did beside each
        # shard, so the assertion is checkable.
        try:
            recorded = runner.read_shard_provenance(
                runner.meta_path_for(results_path)
            )
        except runner.ShardWithheld as exc:
            # A SIDECAR NOBODY COULD READ IS NOT A SIDECAR-LESS SHARD. Both used
            # to answer None, so a truncated or unreadable one joined on the
            # unverified path where every assertion below compares nothing --
            # the fault passed as agreement about a configuration nobody read.
            raise SystemExit(
                f"{exc}. Its verdicts cannot be joined under an unverified "
                "configuration: repair or remove the sidecar, or rescore the "
                "shard."
            ) from exc
        if recorded is None:
            # Shards scored before the sidecar existed. Joined, because they are
            # valid, but the manifest says the configuration was unverified
            # rather than confirmed.
            unrecorded += 1
            recorded = {}
        for key, claimed in asserted.items():
            # served_model_id alone is skipped when unset: it selects the
            # isotonic and an unset one means "resolve none", not an assertion
            # about the run. Every other key asserts, None included.
            if key == "served_model_id" and claimed is None:
                continue
            if key in recorded and recorded[key] != claimed:
                raise SystemExit(
                    f"{results_path.name} was scored with {key}="
                    f"{recorded[key]!r}, but this run asserts {claimed!r}. "
                    "The belief profile and the isotonic are both keyed on "
                    "that, so the table would be computed — and its manifest "
                    "written — for a run that never happened."
                )
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

    if unrecorded:
        # THE MANIFEST IS NOT WHERE AN OPERATOR LOOKS FIRST. Every shard the 60M
        # run has published predates the sidecar, so --model/--variant/
        # --stmt-hash-filter are unverifiable assertions over that part of the
        # table -- including the case the --stmt-hash-filter help describes as
        # refused, which is refusable only where a sidecar recorded the digest.
        print(
            f"  [provenance] {unrecorded:,} of {stats['n_shards']:,} joined "
            "shards recorded no sidecar: --model, --variant and "
            "--stmt-hash-filter are UNVERIFIED for them "
            "(manifest: n_shards_without_provenance)",
            flush=True,
        )

    # A run that found no shard results at all is a configuration error wearing
    # a successful exit code -- most often a --limit the scoring run used and
    # this one was not told about. An empty table is never a legitimate answer.
    if stats["n_shards"] == 0:
        raise SystemExit(
            f"no shard results were read ({stats['n_missing_results']} input "
            f"shards had no matching output in {results_dir}). If the scoring run "
            "used --limit, pass the same --limit here; the filenames carry it."
        )

    # The guard above counts shards whose FILE was found, so a join that read
    # every file and kept nothing -- a scoring window where every cell finalized
    # as "error", or results from a different preparation generation -- fell
    # straight through it and published `{}` with exit 0. An empty table is not
    # a harmless empty file: `build_table` omits unscored statements precisely
    # because INDRA's Statement.from_json defaults a missing belief to 1.0, so a
    # table with nothing in it makes the whole corpus read as certainly true.
    if stats["n_statements"] == 0:
        raise SystemExit(
            f"{stats['n_shards']} shard results were read and NOT ONE statement "
            f"survived the join ({stats['n_unscored']:,} evidences had no usable "
            "cell). Publishing an empty table would leave every corpus statement "
            "to default to belief 1.0 downstream."
        )

    # PARTIAL COVERAGE READS AS CERTAINTY. The two guards above refuse only the
    # empty cases, so 999 of 1000 shards failing to resolve published a table
    # over 0.1% of the corpus with exit 0 -- and by the argument the guard above
    # already makes (Statement.from_json defaults a missing belief to 1.0), the
    # other 99.9% then read as certainly true rather than as unscored.
    coverage = stats["n_shards"] / len(shards)
    if coverage < args.min_shard_coverage:
        raise SystemExit(
            f"only {stats['n_shards']:,} of {len(shards):,} input shards joined "
            f"({coverage:.1%}), below --min-shard-coverage "
            f"{args.min_shard_coverage:.1%}; {stats['n_missing_results']} had no "
            "matching output. Every statement outside those shards defaults to "
            "belief 1.0 downstream. Pass --min-shard-coverage explicitly to "
            "build over a partial scoring run."
        )

    # An isotonic that was registered but never actually applied produces a table
    # indistinguishable from a calibrated one, with a manifest naming the curve.
    if calibration is not None and stats["n_weighted"] == 0:
        # THE REASON TRAVELS WITH THE REFUSAL. `apply_weights` records the first
        # exception so "nothing weighted" is a diagnosis rather than a mystery,
        # and it was written only to the manifest -- which this refusal aborts
        # before writing, so the one run that needs the reason is the one run
        # that discards it.
        why = stats.get("first_weight_error")
        raise SystemExit(
            f"an isotonic is registered but NOT ONE row was weighted "
            f"({stats['n_weight_failed']} failed, {stats['n_null_margin']} carried "
            "no margin). Every belief here is verdict-weighted; publishing it "
            "under a calibrated manifest would misdescribe the whole table."
            + (f" The first failure was: {why}" if why else "")
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
        "margin_route": margin_route,
        "isotonic": path.name if path else None,
        "served_model_id": args.served_model_id,
        "stmt_hash_filter": args.stmt_hash_filter,
        "shard_coverage": coverage,
        # How many joined shards could not confirm the configuration asserted
        # above, because they were scored before the runner wrote a sidecar.
        # An absent sidecar is never read as agreement -- the assertion loop
        # skips a key that is not recorded -- so this number is the size of the
        # UNVERIFIED part of the table, not a count of confirmations.
        "n_shards_without_provenance": unrecorded,
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
