#!/usr/bin/env python3
"""Reasoning on vs off, on the SAME 33,361 executions of the 2023 paper panel.

The shipped comparison ran four reading models with the provider's
chain-of-thought on AND a prompt that told the model to deliberate in its answer.
The 2026-07-31 re-run removed BOTH — provider CoT and prompt scaffolding — and
re-scored the identical corpus, so every row pairs one-to-one on
``(stmt_i, evidence_i)``.  This script measures what that removal bought and cost.

WHY THE NUMBERS ARE COMPUTED HERE AND NOT BY ``model-bundle``
--------------------------------------------------------------
They cannot be.  ``comparison/llm.py`` derives each pair's EXPECTED provider-call
topology from the shared execution map, where 17,235 of 33,361 pairs declare a
``relation_nature`` sub-call.  The verdict-only substrate deliberately removed
that call, so ``model-bundle`` rejects all four arms at their first ``[Complex]``
pair with ``final provider-call topology differs``.  That is a genuine gap
between the run and the frozen bundler, it is recorded in the artifact as
``bundler_status``, and it is why the verdict-only side carries no bundle sha.

WHAT THIS SCRIPT PROVES BEFORE IT MEASURES ANYTHING
----------------------------------------------------
1. The thinking side's raw attempts are read from the path each SHIPPED bundle
   manifest declares, and the file's sha256 must equal the manifest's declared
   digest.  The verdict-only side is read from the path its own frozen run plan
   declares.  Neither is a convenience copy.
2. Exactly one scored row exists per ``(stmt_i, evidence_i)`` on both sides, and
   both sides cover the same pair universe.  Retry predecessors are errors and
   carry no verdict, so "scored" already selects the final row.
3. The thinking side's recovered verdicts, pushed back through
   ``statement_belief`` with the run's own ``aggregation.json`` priors, must
   reproduce the SHIPPED ``all_source_predictions.jsonl`` and
   ``reader_predictions.jsonl`` EXACTLY (max |Δ| == 0), and the resulting
   AP / AUROC / Brier / ECE / confusion must equal the SHIPPED
   ``indra_belief_comparison_metrics.json`` values exactly.
   If any of that fails there is no figure: it would mean the verdicts driving
   this comparison are not the ones behind the numbers /paper already reports.
4. Both runs' ``aggregation.json`` must be byte-identical.  A non-null
   ``reader_profile`` on one side only would put a calibrated reader against a
   hard gate and manufacture an aggregation confound out of nothing.

METRICS.  Average precision and AUROC come from scikit-learn, which is asserted
equal to the shipped estimates on the thinking side before anything else runs, so
the fast path cannot drift from the estimator the rest of /paper uses.  Both are
tie-aware.  Trapezoidal PR-AUC is carried but never the headline -- it inflates
most in exactly this regime.  ECE uses the shipped frozen bin edges.

BOOTSTRAP.  10,000 fold-stratified resamples, seed 20260717, ONE shared index
vector per resample across every arm and BOTH sides, so each delta is paired at
the statement.  Mirrors the design in ``compare_paper_literal_vs_llms.py``.

COST.  The thinking side's cost is READ off the shipped metrics artifact's own
accounting block.  The verdict-only side has no bundle, so it is summed from its
spend WAL by the same rule -- provider-reported settlements are the lower bound,
conservative reservations are the gap to the upper.

Run::

    PYTHONPATH=src python scripts/compute_reasoning_ablation.py \
      --out-dir data/results/reasoning_ablation_20260805
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from indra_belief.metrics import confusion_pr  # noqa: E402
from indra_belief.statement_belief import statement_belief  # noqa: E402

ARTIFACT_KIND = "indra_reasoning_ablation"

# The five reading sources the paper's reader-only panel is built from.
READER_SOURCES = frozenset({"reach", "sparser", "medscan", "rlimsp", "trips"})

RESAMPLES = 10_000
SEED = 20260717

CORPUS = "data/corpora/indra_paper_unique_pairs_20260717_statements.json"
SHIPPED_METRICS = "data/results/indra_belief_comparison_metrics.json"
# The page's HEADLINE metric artifact. Its thinking-side error-class F1 is what
# this run's deliberating arm must reproduce, cut for cut.
SHIPPED_ERROR_F1 = "data/results/indra_paper_literal_models_20260724/statement_error_f1.json"
THINKING_AGGREGATION = "data/comparison/aggregation.json"
VERDICT_ONLY_AGGREGATION = "data/comparison_verdict_only/aggregation.json"
VERDICT_ONLY_PLAN = "data/comparison_verdict_only/run_plan.json"

PANELS = {
    "paper_all_source": {
        "gold": "data/results/indra_paper_comparison_gold_20260717/paper_released_gold.jsonl",
        "predictions": "all_source_predictions.jsonl",
        "display": "all-source panel",
    },
    "paper_readers": {
        "gold": "data/results/indra_paper_comparison_gold_20260717/paper_reader_eligible_released_gold.jsonl",
        "predictions": "reader_predictions.jsonl",
        "display": "five-reader panel",
    },
}

# Each arm pairs one SHIPPED thinking bundle with one verdict-only plan action.
# `arm_id` is the frozen join key; `display` is the only string a reader sees.
ARMS = [
    {
        "arm_id": "gemma_4_26b",
        "display": "Gemma 4 26B",
        "shipped_arm_id": "llm_gemma_4_26b",
        "bundle_dir": "data/comparison/models/gemma_4_26b",
        "verdict_only_action": "gemma_26b_vo_primary",
    },
    {
        "arm_id": "gemma_4_31b",
        "display": "Gemma 4 31B",
        "shipped_arm_id": "llm_gemma_4_31b",
        "bundle_dir": "data/comparison/models/gemma_4_31b",
        "verdict_only_action": "gemma_31b_vo_primary",
    },
    {
        "arm_id": "glm_5",
        "display": "GLM 5",
        "shipped_arm_id": "llm_glm_5",
        "bundle_dir": "data/comparison/models/glm_5",
        "verdict_only_action": "glm_5_vo_primary",
    },
    {
        "arm_id": "gemma_4_e2b",
        "display": "Gemma 4 E2B",
        "shipped_arm_id": "llm_gemma_4_e2b",
        "bundle_dir": "data/comparison/models/gemma_4_e2b",
        "verdict_only_action": "e2b_vo_primary",
    },
]

# Our arm key -> the key the SHIPPED error-F1 artifact addresses the same model by.
ERROR_F1_KEYS = {
    "gemma_4_26b": "gemma-4-26b",
    "gemma_4_31b": "gemma-4-31b",
    "glm_5": "glm-5",
    "gemma_4_e2b": "gemma-4-e2b",
}

# What `model-bundle` says when it is pointed at the verdict-only run, verbatim,
# with the reason it says it. Recorded as DATA so the figure cannot claim the
# verdict-only side passed a gate it never reached.
BUNDLER_STATUS = {
    "state": "rejected",
    "command": "python -m indra_belief.comparison model-bundle --inputs data/comparison_verdict_only/inputs.json",
    "error": "final provider-call topology differs",
    "cause": (
        "comparison/llm.py derives each pair's expected provider-call topology from the "
        "shared execution map, which declares a relation_nature sub-call for 17,235 of "
        "33,361 pairs. The verdict-only substrate removed that call, so every [Complex] "
        "pair mismatches and all four arms are rejected at their first one."
    ),
    "consequence": (
        "The verdict-only side carries no bundle digest. Its statement probabilities are "
        "computed here, by the same statement_belief entry point and the same priors, and "
        "the thinking side is recomputed alongside it and gated against the shipped bundle "
        "so both sides are known to come from one code path."
    ),
}


class GateError(RuntimeError):
    """A precondition that makes the measurement meaningless if it does not hold."""


def fail(context: str, message: str):
    raise GateError(f"{context}: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    return {
        "path": str(path.relative_to(REPO)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------
# recovery
# --------------------------------------------------------------------------

def recover_rows(path: Path, arm_id: str, side: str) -> tuple[dict, dict, str]:
    """Stream one raw attempts file -> {(stmt_i, evidence_i): row}, census, sha256.

    Selection follows ``comparison/llm.py::_validate_raw`` exactly, because that
    is the rule the shipped bundles were built under: rows for a pair appear in
    append order, THE LAST ONE IS FINAL, and it must be scored with a real
    verdict.  A predecessor may be an error OR a scored abstention -- a scored row
    carrying a null verdict, which the E2B run does emit.  Treating "scored" alone
    as final would pick an abstention over the answer that replaced it.
    """
    digest = hashlib.sha256()
    ordered: dict[tuple[int, int], list[dict]] = {}
    census: Counter = Counter()
    with path.open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            if not raw.strip():
                continue
            record = json.loads(raw)
            census["raw_rows"] += 1
            key = (int(record["stmt_i"]), int(record["evidence_i"]))
            ordered.setdefault(key, []).append(record)

    by_key: dict[tuple[int, int], dict] = {}
    for key, rows in ordered.items():
        final = rows[-1]
        if final.get("row_status") != "scored" or final.get("verdict") not in {
            "correct",
            "incorrect",
        }:
            fail(f"{side}[{arm_id}]", f"pair {key} lacks one final scored verdict")
        for predecessor in rows[:-1]:
            if predecessor.get("row_status") != "error" and predecessor.get("verdict") is not None:
                fail(f"{side}[{arm_id}]", f"pair {key} retry predecessor is not an error or abstention")
            census[
                "retry_error_rows"
                if predecessor.get("row_status") == "error"
                else "retry_abstention_rows"
            ] += 1
        by_key[key] = {
            "source_api": str(final.get("source_api") or "").casefold(),
            "verdict": final["verdict"],
            "confidence": final.get("confidence"),
            "tier": final.get("tier"),
            "evidence_json_sha256": final.get("evidence_json_sha256"),
        }
    census["scored_pairs"] = len(by_key)
    return by_key, dict(census), digest.hexdigest()


def predictions(rows: dict, statements: list, priors: dict) -> dict[str, dict[str, float]]:
    """Roll per-evidence verdicts up to statement belief, both panels.

    Mirrors ``comparison/llm.py::_predictions`` field for field: same canonical
    entry point, same priors, same within-source de-dup, same hard gate.
    """
    out: dict[str, dict[str, float]] = {"paper_all_source": {}, "paper_readers": {}}
    for stmt_i, statement in enumerate(statements):
        all_evidence, reader_evidence = [], []
        for evidence_i, evidence in enumerate(statement["evidence"]):
            row = rows[(stmt_i, evidence_i)]
            source = str(evidence.get("source_api") or "").casefold()
            if row["source_api"] != source:
                fail("predictions", f"pair {(stmt_i, evidence_i)} disagrees on source_api")
            measurement = {
                "source_api": source,
                "verdict": row["verdict"],
                "confidence": row["confidence"],
                "tier": row["tier"],
                "evidence_text": evidence.get("text"),
                "evidence_hash": row["evidence_json_sha256"],
            }
            all_evidence.append(measurement)
            if source in READER_SOURCES:
                reader_evidence.append(measurement)
        belief = statement_belief(all_evidence, priors=dict(priors), dedup=True, soft=None)
        out["paper_all_source"][statement["id"]] = float(belief.belief)
        if reader_evidence:
            reader = statement_belief(reader_evidence, priors=dict(priors), dedup=True, soft=None)
            out["paper_readers"][statement["id"]] = float(reader.belief)
    return out


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def expected_calibration_error(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray, edges: np.ndarray
) -> float:
    """ECE over the shipped frozen bins.

    Reimplemented rather than imported so this script depends on no private
    symbol; gated by equality against the shipped estimate on the thinking side.
    """
    index = np.clip(np.searchsorted(edges[1:-1], scores, side="right"), 0, len(edges) - 2)
    total = float(np.sum(weights))
    error = 0.0
    for bin_id in range(len(edges) - 1):
        take = index == bin_id
        mass = float(np.sum(weights[take]))
        if mass <= 0:
            continue
        mean_score = float(np.sum(weights[take] * scores[take]) / mass)
        mean_label = float(np.sum(weights[take] * labels[take]) / mass)
        error += (mass / total) * abs(mean_score - mean_label)
    return error


def confusion(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold
    tp = int(np.sum(predicted & labels))
    fp = int(np.sum(predicted & ~labels))
    fn = int(np.sum(~predicted & labels))
    tn = int(np.sum(~predicted & ~labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / len(labels),
    }


def point_metrics(
    labels: np.ndarray, scores: np.ndarray, edges: np.ndarray, threshold: float
) -> dict:
    weights = np.ones(len(labels), dtype=float)
    return {
        "average_precision": float(average_precision_score(labels.astype(int), scores)),
        "auroc": float(roc_auc_score(labels.astype(int), scores)),
        "brier": float(np.mean((scores - labels.astype(float)) ** 2)),
        "ece": expected_calibration_error(labels.astype(float), scores, weights, edges),
        "confusion": confusion(labels, scores, threshold),
    }


def error_operating_point(scores: np.ndarray, is_error: np.ndarray, tau: float) -> dict:
    """One cut, described on the ERROR class, through the repo's own P/R/F1.

    ``flagged = belief < tau`` and the confusion goes through
    ``indra_belief.metrics.confusion_pr`` -- the same decision rule and the same
    single definition ``scripts/compute_statement_error_f1.py`` uses, so this
    cannot drift from the number /paper publishes.
    """
    flagged = scores < tau
    err = confusion_pr(zip(is_error.tolist(), flagged.tolist()))
    return {
        "tau": float(tau),
        "flagged": int(flagged.sum()),
        "error_precision": err["p"],
        "error_recall": err["r"],
        "error_f1": err["f1"],
        "tp": err["tp"],
        "fp": err["fp"],
        "fn": err["fn"],
        "tn": err["tn"],
    }


def best_error_f1_threshold(scores: np.ndarray, is_error: np.ndarray) -> float:
    """The side's own best-error-F1 cut, ties broken to the SMALLEST such cut.

    Candidates are the side's OWN DISTINCT SCORES, not midpoints between them --
    the shipped rule, quoted: "of the arm's own distinct scores, the one whose
    flag set {belief < tau} maximises error-class F1".  A midpoint grid finds the
    same F1 on a plateau but names a different tau, which would not reconcile.
    """
    best_tau, best_f1 = None, -1.0
    for tau in np.unique(scores):
        f1 = confusion_pr(zip(is_error.tolist(), (scores < tau).tolist()))["f1"]
        if f1 > best_f1:
            best_tau, best_f1 = float(tau), f1
    if best_tau is None or best_f1 <= 0.0:
        fail("error_f1", "no cut separates this panel at all")
    return best_tau


def error_f1_at(is_error: np.ndarray, flagged: np.ndarray) -> float:
    """Vectorised error-class F1 for the bootstrap inner loop.

    Asserted equal to ``confusion_pr`` on the full panel before any resample runs,
    so the fast path cannot drift from the definition above.
    """
    tp = float(np.sum(flagged & is_error))
    fp = float(np.sum(flagged & ~is_error))
    fn = float(np.sum(~flagged & is_error))
    if tp <= 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def fold_stratified_indices(folds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One resample: draw within each frozen fold, preserving its size."""
    picks = []
    for fold in np.unique(folds):
        members = np.flatnonzero(folds == fold)
        picks.append(rng.choice(members, size=len(members), replace=True))
    return np.concatenate(picks)


def standing(low: float, high: float) -> str:
    """Where an interval sits relative to zero. Read off the ENDPOINTS only."""
    if low > 0:
        return "ahead"
    if high < 0:
        return "behind"
    return "not-significant"


# --------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------

def sum_settled_spend(path: Path, run_id: str) -> dict:
    """Sum one action's settled calls from its WAL, split by accounting basis."""
    marker = '"event":"call_settled"'
    provider = Decimal(0)
    conservative = Decimal(0)
    counts = Counter()
    input_tokens = output_tokens = 0
    by_kind: Counter = Counter()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if marker not in line:
                continue
            row = json.loads(line)
            if row.get("run_id") != run_id:
                continue
            by_kind[row.get("kind")] += 1
            cost = Decimal(str(row["settled_cost_usd"]))
            if row.get("accounting_basis") == "provider_reported_usage":
                provider += cost
                counts["provider_measured_calls"] += 1
                usage = row.get("provider_usage") or {}
                input_tokens += int(usage.get("input_tokens") or 0)
                output_tokens += int(usage.get("output_tokens") or 0)
            else:
                conservative += cost
                counts["conservative_calls"] += 1
    return {
        "basis": "wal_settlement_sum",
        "inference_usd_lower": float(provider),
        "inference_usd_upper": float(provider + conservative),
        "provider_measured_calls": counts["provider_measured_calls"],
        "conservative_calls": counts["conservative_calls"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "calls_by_kind": dict(by_kind),
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--skip-verdict-only-cost",
        action="store_true",
        help="skip the verdict-only WAL sweep (~10 GB) and emit no cost block for it",
    )
    args = parser.parse_args()

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- gate: the two runs must aggregate identically -------------------
    thinking_aggregation = (REPO / THINKING_AGGREGATION).read_bytes()
    verdict_only_aggregation = (REPO / VERDICT_ONLY_AGGREGATION).read_bytes()
    if thinking_aggregation != verdict_only_aggregation:
        fail(
            "aggregation",
            "the two runs' aggregation.json differ, so a rollup difference would be "
            "indistinguishable from a reading difference",
        )
    aggregation = json.loads(thinking_aggregation)
    if aggregation.get("reader_profile") is not None:
        fail("aggregation", "expected the hard gate on both sides")
    priors = {key: tuple(value) for key, value in aggregation["priors"].items()}

    error_f1_artifact = json.loads((REPO / SHIPPED_ERROR_F1).read_text())
    shipped_error_f1 = {
        arm["key"]: arm["operating_point"]
        for arm in error_f1_artifact["arms"]
        if arm.get("role") == "reader-gate" and isinstance(arm.get("operating_point"), dict)
    }
    error_rules = {
        "decision_rule": error_f1_artifact["decision_rule"],
        "threshold_rule": error_f1_artifact["threshold_rule"],
        "oracle_disclosure": error_f1_artifact["oracle_disclosure"],
    }

    shipped = json.loads((REPO / SHIPPED_METRICS).read_text())
    edges = np.array(shipped["provenance"]["calibration_bin_edges"], dtype=float)
    shipped_arms = {
        (substrate["substrate_id"], arm["arm_id"]): arm
        for substrate in shipped["substrates"]
        for arm in substrate["arms"]
    }

    statements = json.loads((REPO / CORPUS).read_text())
    plan = json.loads((REPO / VERDICT_ONLY_PLAN).read_text())
    plan_actions = {action["id"]: action for action in plan["actions"]}
    plan_stages = {stage["id"]: stage for stage in plan["stages"]}

    gold: dict[str, dict] = {}
    for panel, spec in PANELS.items():
        rows = load_jsonl(REPO / spec["gold"])
        gold[panel] = {
            "ids": [row["statement_id"] for row in rows],
            "label": np.array([bool(row["label"]) for row in rows]),
            "fold": np.array([int(row["fold_id"]) for row in rows]),
        }

    arms_out = []
    aligned: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for spec in ARMS:
        arm_id = spec["arm_id"]
        print(f"[{arm_id}] recovering …", file=sys.stderr, flush=True)

        # --- thinking side: path and digest come from the SHIPPED bundle ---
        manifest_path = REPO / spec["bundle_dir"] / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        declared = manifest["implementation"]["notes"]["inputs"].get("raw_attempts")
        if not declared:
            fail(f"thinking[{arm_id}]", "bundle manifest declares no raw_attempts input")
        thinking_path = (manifest_path.parent / declared["path"]).resolve()
        thinking_rows, thinking_census, thinking_digest = recover_rows(
            thinking_path, arm_id, "thinking"
        )
        if thinking_digest != declared["sha256"]:
            fail(
                f"thinking[{arm_id}]",
                f"raw_attempts sha256 {thinking_digest} != manifest-declared {declared['sha256']}",
            )

        # --- verdict-only side: path comes from its own frozen run plan ----
        action = plan_actions.get(spec["verdict_only_action"])
        if action is None:
            fail(f"verdict_only[{arm_id}]", f"plan declares no action {spec['verdict_only_action']!r}")
        vo_path = REPO / action["output"]["path"]
        vo_rows, vo_census, vo_digest = recover_rows(vo_path, arm_id, "verdict_only")

        if set(thinking_rows) != set(vo_rows):
            fail(arm_id, "the two runs do not cover the same execution pair universe")

        # --- gate: thinking side must rebuild the SHIPPED bundle exactly ---
        thinking_predictions = predictions(thinking_rows, statements, priors)
        reconciliation = {}
        for panel, panel_spec in PANELS.items():
            shipped_rows = {
                row["statement_id"]: float(row["probability_correct"])
                for row in load_jsonl(REPO / spec["bundle_dir"] / panel_spec["predictions"])
            }
            mine = thinking_predictions[panel]
            if set(mine) != set(shipped_rows):
                fail(f"reconciliation[{arm_id}/{panel}]", "statement id sets differ")
            diffs = [abs(mine[key] - shipped_rows[key]) for key in shipped_rows]
            max_abs = max(diffs) if diffs else 0.0
            if max_abs != 0.0:
                fail(
                    f"reconciliation[{arm_id}/{panel}]",
                    f"recovered verdicts do not rebuild the shipped bundle (max |Δ| = {max_abs:g})",
                )
            reconciliation[panel] = {
                "n_statements": len(shipped_rows),
                "n_exact": sum(1 for value in diffs if value == 0.0),
                "max_abs_diff": max_abs,
            }

        vo_predictions = predictions(vo_rows, statements, priors)

        # --- evidence grain -------------------------------------------------
        transitions: Counter = Counter()
        llm_tier = Counter()
        for key, thinking_row in thinking_rows.items():
            vo_row = vo_rows[key]
            transitions[f"{thinking_row['verdict']}_to_{vo_row['verdict']}"] += 1
            if thinking_row["tier"] == "llm_comprehension" == vo_row["tier"]:
                llm_tier["n"] += 1
                if thinking_row["verdict"] != vo_row["verdict"]:
                    llm_tier["flips"] += 1
                    llm_tier[
                        "to_incorrect" if vo_row["verdict"] == "incorrect" else "to_correct"
                    ] += 1

        evidence_grain = {
            "n_executions": len(thinking_rows),
            "reasoning": {
                "correct": sum(1 for r in thinking_rows.values() if r["verdict"] == "correct"),
                "incorrect": sum(1 for r in thinking_rows.values() if r["verdict"] == "incorrect"),
            },
            "verdict_only": {
                "correct": sum(1 for r in vo_rows.values() if r["verdict"] == "correct"),
                "incorrect": sum(1 for r in vo_rows.values() if r["verdict"] == "incorrect"),
            },
            "transitions": dict(transitions),
            "llm_tier": {
                "n": llm_tier["n"],
                "flips": llm_tier["flips"],
                "to_correct": llm_tier["to_correct"],
                "to_incorrect": llm_tier["to_incorrect"],
                "agreement": 1 - llm_tier["flips"] / llm_tier["n"] if llm_tier["n"] else 0.0,
            },
        }

        # --- statement grain, both panels -----------------------------------
        aligned[arm_id] = {}
        panels_out = {}
        for panel, panel_spec in PANELS.items():
            ids = gold[panel]["ids"]
            labels = gold[panel]["label"]
            shipped_arm = shipped_arms[(panel, spec["shipped_arm_id"])]
            threshold = float(shipped_arm["metrics"]["threshold"]["value"])
            if shipped_arm["metrics"]["threshold"]["operator"] != "greater_than_or_equal":
                fail(f"threshold[{arm_id}/{panel}]", "unexpected threshold operator")

            missing = [key for key in ids if key not in vo_predictions[panel]]
            if missing:
                fail(f"{arm_id}/{panel}", f"{len(missing)} gold statements have no prediction")

            thinking_scores = np.array([thinking_predictions[panel][key] for key in ids])
            vo_scores = np.array([vo_predictions[panel][key] for key in ids])
            aligned[arm_id][panel] = {"thinking": thinking_scores, "verdict_only": vo_scores}

            thinking_point = point_metrics(labels, thinking_scores, edges, threshold)
            vo_point = point_metrics(labels, vo_scores, edges, threshold)

            # --- gate: the recomputed thinking side must equal the shipped one
            published = {
                "average_precision": shipped_arm["metrics"]["pooled_average_precision"]["estimate"],
                "auroc": shipped_arm["metrics"]["auroc"]["estimate"],
                "brier": shipped_arm["metrics"]["brier"]["estimate"],
                "ece": shipped_arm["metrics"]["calibration"]["ece"]["estimate"],
            }
            for name, want in published.items():
                got = thinking_point[name]
                if abs(got - want) > 1e-12:
                    fail(
                        f"shipped_parity[{arm_id}/{panel}]",
                        f"recomputed {name} {got!r} != shipped {want!r}",
                    )
            shipped_confusion = shipped_arm["metrics"]["threshold"]["confusion"]
            for cell in ("tp", "fp", "fn", "tn"):
                if thinking_point["confusion"][cell] != shipped_confusion[cell]:
                    fail(
                        f"shipped_parity[{arm_id}/{panel}]",
                        f"recomputed confusion {cell} differs from shipped",
                    )

            # --- the ERROR class: the metric this page actually leads on ------
            # Two threshold rules, both real, and the ablation reads differently
            # under each:
            #   own_cut      each side at its OWN full-panel best-error-F1 tau --
            #                the shipped headline rule, and an ORACLE cut.
            #   deployed_cut both sides at the DELIBERATING side's tau -- what a
            #                curator who swapped the model and left the cutoff
            #                alone would actually get.
            is_error = ~labels
            tau_reasoning = best_error_f1_threshold(thinking_scores, is_error)
            tau_verdict_only = best_error_f1_threshold(vo_scores, is_error)
            error_class = {
                "own_cut": {
                    "reasoning": error_operating_point(thinking_scores, is_error, tau_reasoning),
                    "verdict_only": error_operating_point(vo_scores, is_error, tau_verdict_only),
                },
                "deployed_cut": {
                    "tau": tau_reasoning,
                    "reasoning": error_operating_point(thinking_scores, is_error, tau_reasoning),
                    "verdict_only": error_operating_point(vo_scores, is_error, tau_reasoning),
                },
            }
            # The fast path used inside the bootstrap must agree with the
            # definition above on the full panel, for every cut it will be asked
            # about, before a single resample runs.
            for scores, tau in (
                (thinking_scores, tau_reasoning),
                (vo_scores, tau_verdict_only),
                (vo_scores, tau_reasoning),
            ):
                fast = error_f1_at(is_error, scores < tau)
                exact = error_operating_point(scores, is_error, tau)["error_f1"]
                if abs(fast - exact) > 1e-12:
                    fail(f"error_f1[{arm_id}/{panel}]", "the bootstrap fast path differs from confusion_pr")

            # --- gate: the deliberating side must BE the published error-F1 ----
            shipped_key = ERROR_F1_KEYS.get(arm_id)
            if panel == "paper_all_source" and shipped_key is not None:
                published = shipped_error_f1.get(shipped_key)
                if published is None:
                    fail(f"error_f1[{arm_id}]", f"{SHIPPED_ERROR_F1} declares no arm {shipped_key!r}")
                own = error_class["own_cut"]["reasoning"]
                for field in ("tau", "error_f1"):
                    if abs(own[field] - float(published[field])) > 1e-12:
                        fail(
                            f"error_f1_parity[{arm_id}]",
                            f"recomputed {field} {own[field]!r} != published {published[field]!r}",
                        )
                error_class["shipped_parity_verified"] = True

            panels_out[panel] = {
                "n_evaluable": len(ids),
                "n_positive": int(np.sum(labels)),
                "n_negative": int(np.sum(~labels)),
                "threshold": threshold,
                "reasoning": thinking_point,
                "verdict_only": vo_point,
                "error_class": error_class,
                "shipped_parity_verified": True,
            }

        arms_out.append(
            {
                "arm_id": arm_id,
                "display": spec["display"],
                "shipped_arm_id": spec["shipped_arm_id"],
                "reasoning": {
                    "served_model": manifest["implementation"]["notes"]["served_model"],
                    "provider_model_id": manifest["implementation"]["notes"]["provider_model_id"],
                    "run_id": manifest["run_id"],
                    "raw_attempts": {
                        "path": str(thinking_path.relative_to(REPO)),
                        "sha256": thinking_digest,
                        "sha256_matches_bundle_manifest": True,
                    },
                    "census": thinking_census,
                },
                "verdict_only": {
                    "served_model": plan_stages[action["stage"]]["model"],
                    "provider_model_id": plan_stages[action["stage"]]["provider_model_id"],
                    "run_id": action["run_id"],
                    "raw_attempts": {
                        "path": str(vo_path.relative_to(REPO)),
                        "sha256": vo_digest,
                        "sha256_matches_bundle_manifest": False,
                    },
                    "census": vo_census,
                },
                "reconciliation": reconciliation,
                "evidence_grain": evidence_grain,
                "panels": panels_out,
            }
        )

    # --- paired bootstrap: one shared index vector per resample -------------
    print("[bootstrap] 10,000 fold-stratified resamples …", file=sys.stderr, flush=True)
    METRICS = ("average_precision", "auroc", "error_f1_own_cut", "error_f1_deployed_cut")
    deltas: dict[str, dict[str, dict[str, list[float]]]] = {
        arm["arm_id"]: {panel: {metric: [] for metric in METRICS} for panel in PANELS}
        for arm in arms_out
    }
    for panel in PANELS:
        labels = gold[panel]["label"]
        folds = gold[panel]["fold"]
        is_error = ~labels
        rng = np.random.default_rng(SEED)
        for _ in range(RESAMPLES):
            index = fold_stratified_indices(folds, rng)
            drawn_labels = labels[index]
            # A resample with no error at all has no error-class recall to speak
            # of, and one with no correct statement has no ranking; both are
            # skipped, exactly as the shipped error-F1 bootstrap skips them.
            if drawn_labels.all() or not drawn_labels.any():
                continue
            truth = drawn_labels.astype(int)
            drawn_errors = is_error[index]
            for arm in arms_out:
                pair = aligned[arm["arm_id"]][panel]
                thinking = pair["thinking"][index]
                verdict_only = pair["verdict_only"][index]
                block = deltas[arm["arm_id"]][panel]
                block["average_precision"].append(
                    average_precision_score(truth, verdict_only)
                    - average_precision_score(truth, thinking)
                )
                block["auroc"].append(
                    roc_auc_score(truth, verdict_only) - roc_auc_score(truth, thinking)
                )
                # Taus are the FULL-PANEL values and are NOT refit per resample:
                # the interval is on the metric at fixed cuts. Same rule as the
                # shipped error-F1 bootstrap, quoted in its `bootstrap_design`.
                cuts = arm["panels"][panel]["error_class"]
                own_reasoning = cuts["own_cut"]["reasoning"]["tau"]
                own_verdict_only = cuts["own_cut"]["verdict_only"]["tau"]
                deployed = cuts["deployed_cut"]["tau"]
                block["error_f1_own_cut"].append(
                    error_f1_at(drawn_errors, verdict_only < own_verdict_only)
                    - error_f1_at(drawn_errors, thinking < own_reasoning)
                )
                block["error_f1_deployed_cut"].append(
                    error_f1_at(drawn_errors, verdict_only < deployed)
                    - error_f1_at(drawn_errors, thinking < deployed)
                )

    for arm in arms_out:
        for panel in PANELS:
            measured = arm["panels"][panel]
            cuts = measured["error_class"]
            points = {
                "average_precision": (
                    measured["verdict_only"]["average_precision"]
                    - measured["reasoning"]["average_precision"]
                ),
                "auroc": measured["verdict_only"]["auroc"] - measured["reasoning"]["auroc"],
                "error_f1_own_cut": (
                    cuts["own_cut"]["verdict_only"]["error_f1"]
                    - cuts["own_cut"]["reasoning"]["error_f1"]
                ),
                "error_f1_deployed_cut": (
                    cuts["deployed_cut"]["verdict_only"]["error_f1"]
                    - cuts["deployed_cut"]["reasoning"]["error_f1"]
                ),
            }
            block = {}
            for metric in METRICS:
                draws = np.array(deltas[arm["arm_id"]][panel][metric])
                low = float(np.quantile(draws, 0.025))
                high = float(np.quantile(draws, 0.975))
                block[metric] = {
                    "value": points[metric],
                    "ci95": [low, high],
                    "standing": standing(low, high),
                    "bootstrap_se": float(np.std(draws, ddof=1)),
                    "resamples": RESAMPLES,
                    "valid_resamples": int(len(draws)),
                }
            measured["delta"] = block

    # --- multiplicity: one max-t band over the four models we ran -------------
    # Four models were compared, so a pointwise interval understates how often at
    # least one of them clears zero by chance. The draws share one index vector
    # per resample, so the studentised maximum across models is available from
    # what was already collected. Mirrors the shipped error-F1 artifact's
    # `simultaneous_low` / `simultaneous_high` / `excludes_zero_simultaneous`.
    for panel in PANELS:
        for metric in METRICS:
            family = [arm["arm_id"] for arm in arms_out]
            columns = [np.array(deltas[arm_id][panel][metric]) for arm_id in family]
            width = min(len(column) for column in columns)
            errors = [float(np.std(column, ddof=1)) for column in columns]
            if min(errors) <= 0:
                fail(f"max_t[{panel}/{metric}]", "a degenerate bootstrap standard error")
            studentised = np.max(
                np.abs(
                    np.vstack(
                        [
                            (column[:width] - column[:width].mean()) / error
                            for column, error in zip(columns, errors, strict=True)
                        ]
                    )
                ),
                axis=0,
            )
            critical = float(np.quantile(studentised, 0.95))
            for arm, error in zip(arms_out, errors, strict=True):
                entry = arm["panels"][panel]["delta"][metric]
                low = entry["value"] - critical * error
                high = entry["value"] + critical * error
                entry["simultaneous"] = {
                    "ci95": [low, high],
                    "standing": standing(low, high),
                    "critical_value": critical,
                    "family_size": len(family),
                    "family": family,
                }

    # --- cost ---------------------------------------------------------------
    for arm, spec in zip(arms_out, ARMS, strict=True):
        shipped_cost = shipped_arms[("paper_all_source", spec["shipped_arm_id"])]["cost"]
        arm["reasoning"]["cost"] = {
            "basis": shipped_cost["basis"],
            "inference_usd_lower": shipped_cost["inference_usd_lower"],
            "inference_usd_upper": shipped_cost["inference_usd_upper"],
            "source": "read off the shipped comparison metrics artifact",
        }
        if args.skip_verdict_only_cost:
            arm["verdict_only"]["cost"] = None
            continue
        action = plan_actions[spec["verdict_only_action"]]
        print(f"[{arm['arm_id']}] summing verdict-only spend …", file=sys.stderr, flush=True)
        arm["verdict_only"]["cost"] = sum_settled_spend(
            REPO / action["ledger"]["path"], action["run_id"]
        )

    artifact = {
        "artifact_kind": ARTIFACT_KIND,
        "frozen_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "bundler_status": BUNDLER_STATUS,
        "provenance": {
            "corpus": file_record(REPO / CORPUS),
            "aggregation": file_record(REPO / THINKING_AGGREGATION),
            "aggregation_identical_across_runs": True,
            "shipped_metrics": file_record(REPO / SHIPPED_METRICS),
            "verdict_only_run_plan": file_record(REPO / VERDICT_ONLY_PLAN),
            "statement_belief_implementation": "indra_belief.statement_belief:statement_belief",
            "bootstrap": {
                "resamples": RESAMPLES,
                "seed": SEED,
                "design": (
                    "paired statement bootstrap, stratified independently within each frozen "
                    "fold while preserving its size, one shared index vector across every arm "
                    "and both sides"
                ),
            },
            "estimators": {
                "average_precision": "sklearn.metrics.average_precision_score",
                "auroc": "sklearn.metrics.roc_auc_score",
                "error_f1": "indra_belief.metrics:confusion_pr on the error class",
                "parity": "all asserted equal to the shipped estimates on the reasoning side",
            },
            "error_class": {
                **error_rules,
                "headline": (
                    "error-class F1 is the metric /paper leads on: the panel is 73.2% positive, "
                    "so ranking measures barely move while error catching does"
                ),
                "deployed_cut_rule": (
                    "both sides cut at the DELIBERATING side's own tau -- what a curator who "
                    "swapped the model and left the cutoff alone would get. No oracle is "
                    "available to the verdict-only side under this rule."
                ),
                "shipped_error_f1": file_record(REPO / SHIPPED_ERROR_F1),
            },
            "calibration_bin_edges": [float(value) for value in edges],
        },
        "panels": {
            panel: {
                "display": spec["display"],
                "gold": file_record(REPO / spec["gold"]),
                "n_evaluable": len(gold[panel]["ids"]),
                "n_positive": int(np.sum(gold[panel]["label"])),
                "n_negative": int(np.sum(~gold[panel]["label"])),
            }
            for panel, spec in PANELS.items()
        },
        "arms": arms_out,
    }

    out_path = out_dir / "reasoning_ablation.json"
    out_path.write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n")
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_kind": ARTIFACT_KIND,
                "frozen_at": artifact["frozen_at"],
                "files": {"reasoning_ablation.json": file_record(out_path)},
            },
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {out_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as error:
        print(f"GATE FAILED — {error}", file=sys.stderr)
        raise SystemExit(2)
