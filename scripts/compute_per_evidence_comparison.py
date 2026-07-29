#!/usr/bin/env python3
"""Per-EVIDENCE comparison on the 2023 INDRA paper panel.

Every other figure on /paper is at STATEMENT grain.  This one is at the grain the
system natively operates on: the reader emits one ``correct``/``incorrect``
verdict per ``(statement, evidence)`` pair, and statement belief is a DERIVED
quantity — those per-evidence verdicts pushed through INDRA's noisy-OR.

WHAT THIS SCRIPT PROVES BEFORE IT MEASURES ANYTHING
---------------------------------------------------
1. The per-evidence verdicts are recovered from the per-arm ``raw_attempts``
   file each bundle manifest DECLARES (path + sha256), not from a convenience
   copy.  ``data/comparison/models/<arm>/all_source_attempts.jsonl`` carries NO
   verdict — it is a token/cost execution ledger — so it is not used here.
2. Those recovered verdicts, run back through ``statement_belief`` with the same
   ``aggregation.json`` priors, must reproduce the SHIPPED statement-grain
   ``all_source_predictions.jsonl`` EXACTLY (max |Δ| == 0).  That reconciliation
   is a hard gate: if it fails, the per-evidence rows are not the ones behind the
   statement numbers the rest of the page reports, and there is no figure.
3. The join key is ``(paper_statement_hash, source_hash)``.  ``source_hash``
   alone is NOT unique across the panel (33,101 distinct hashes over 33,361
   pairs), so keying on it alone would silently cross statements.

BASELINES (all constant-within-source by construction, which is the point)
--------------------------------------------------------------------------
* ``indra-default-source-prior`` — INDRA's bundled per-source ``(rand, syst)``
  scored at ONE evidence: ``P(correct) = 1 - (syst_s + rand_s)``.  This is the
  implicit per-evidence model behind the noisy-OR and it is unfitted.
* ``indra-bayes-source-oof`` / ``indra-bayes-subtype-oof`` — INDRA 1.24.0's
  ``BayesianScorer`` refit out-of-fold from this panel's own curation counts,
  read off the shipped fit provenance.  NEITHER IS A PUBLISHED PAPER METHOD; the
  2023 paper publishes no Bayesian or subtype arm.  They are labelled as INDRA
  library code throughout and every arm carries an ``attribution`` string.

The paper's RF / logistic / KNN / SVC rows CANNOT be drawn here at all: their
features (#PMIDs, statement type, promoter, mean evidence length) are statement
level aggregates and the models emit no per-evidence quantity.  That is recorded
as an explicit exclusion rather than approximated.

METRICS.  AUROC and average precision are computed from per-score-bucket
contingency counts in closed form.  Both are tie-aware and both are asserted
equal to ``sklearn.roc_auc_score`` / ``sklearn.average_precision_score`` on the
full sample before any bootstrap runs, so the fast path cannot drift from the
estimator the rest of the page uses.  Trapezoidal PR-AUC is deliberately absent:
the reader's per-evidence score takes at most five distinct values, which is the
regime where trapezoidal interpolation inflates most.

BOOTSTRAP.  10,000 source-stratified resamples, seed 20260717, ONE shared index
vector per resample across every arm (mirrors
``scripts/compare_paper_literal_vs_llms.py``'s fold-stratified design).  Source
stratification makes the per-source strata valid resamples of themselves, so the
pooled and per-source intervals come from the same draws.

Run::

    PYTHONPATH=src python scripts/compute_per_evidence_comparison.py \
      --out-dir data/results/per_evidence_comparison_20260727
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from indra_belief.statement_belief import statement_belief  # noqa: E402

ARTIFACT_KIND = "indra_per_evidence_comparison"
PANEL_ID = "paper_all_source_evidence"
N_BOOT = 10000
SEED = 20260717
# The reader's own decision boundary: verdict "incorrect" maps to score < 0.5.
DECISION_THRESHOLD = 0.5
# Below this, a per-source stratum is reported in the census but never given a
# metric row: an 8-pair stratum produces an interval that spans the axis.
MIN_STRATUM_N = 50
# Metric agreement against sklearn on the full sample. Both closed forms are
# algebraically exact; this is float-noise headroom, not a fudge factor.
SKLEARN_TOLERANCE = 1e-12
# The reaggregation gate. Not a tolerance -- the two paths must agree bit for bit.
REAGGREGATION_TOLERANCE = 0.0

DEFAULTS = {
    "evidence_gold": "data/results/indra_paper_statement_gold_20260717/paper_evidence_adjudication.jsonl",
    "statement_gold": "data/results/indra_paper_statement_gold_20260717/paper_statement_gold.jsonl",
    "fold_gold": "data/results/indra_paper_comparison_gold_20260717/paper_released_gold.jsonl",
    "execution_map": "data/benchmark/indra_paper_unique_pairs_20260717_execution_map.jsonl",
    "corpus": "data/corpora/indra_paper_unique_pairs_20260717_statements.json",
    "models_dir": "data/comparison/models",
    "aggregation": "data/comparison/aggregation.json",
    "default_belief_probs": "data/comparison/models/indra_cogex_hybrid/sources/default_belief_probs.json",
    "bayes_provenance": "data/results/current_indra_bayesian_paper_20260717/current_bayesian_oof_fit_provenance.jsonl",
    "prompt_manifest": "data/comparison/grounding_replay/manifest.json",
    "published_methods": "data/benchmark/indra_paper_2023_published_method_metrics.json",
    "statement_headtohead": "data/results/indra_paper_literal_models_20260724/paper_literal_vs_llms.json",
}

# The SAME model's statement-grain prediction file, so the figure can put both
# grains of one model on one axis. These are read, not recomputed from evidence:
# the point of the bridge is that two INDEPENDENT shipped artifacts agree.
STATEMENT_PREDICTIONS = {
    "llm-gemma-4-e2b": "data/comparison/models/gemma_4_e2b/all_source_predictions.jsonl",
    "llm-gemma-4-26b": "data/comparison/models/gemma_4_26b/all_source_predictions.jsonl",
    "llm-gemma-4-31b": "data/comparison/models/gemma_4_31b/all_source_predictions.jsonl",
    "llm-glm-5": "data/comparison/models/glm_5/all_source_predictions.jsonl",
    "indra-default-source-prior":
        "data/results/current_indra_bayesian_paper_20260717/"
        "current_simple_direct_all_sources_predictions.jsonl",
    "indra-bayes-source-oof":
        "data/results/current_indra_bayesian_paper_20260717/"
        "current_bayesian_source_oof_all_sources_predictions.jsonl",
    "indra-bayes-subtype-oof":
        "data/results/current_indra_bayesian_paper_20260717/"
        "current_bayesian_source_subtype_oof_all_sources_predictions.jsonl",
}

# `point_metrics` keys in the shipped statement head-to-head, for the arms that
# appear in both. Recomputing a number the page already reports is only safe if
# it lands on that number exactly, so this pairing exists to be checked.
HEADTOHEAD_KEY_BY_ARM = {
    "llm-gemma-4-e2b": "Gemma 4 E2B",
    "llm-gemma-4-26b": "Gemma 4 26B",
    "llm-gemma-4-31b": "Gemma 4 31B",
    "llm-glm-5": "GLM-5",
}
STATEMENT_CROSSCHECK_TOLERANCE = 1e-9

# Reader arms. `display` is the on-screen name and is DECOUPLED from `arm`, which
# is the frozen directory join key. Renaming a display name must never move data.
READER_ARMS = [
    ("llm-gemma-4-e2b", "gemma_4_e2b", "Gemma 4 E2B"),
    ("llm-gemma-4-26b", "gemma_4_26b", "Gemma 4 26B"),
    ("llm-gemma-4-31b", "gemma_4_31b", "Gemma 4 31B"),
    ("llm-glm-5", "glm_5", "GLM-5"),
]

BAYES_SOURCE_ARM = "indra_1.24.0_bayesian_source_oof_all_sources"
BAYES_SUBTYPE_ARM = "indra_1.24.0_bayesian_source_subtype_oof_all_sources"

# Attribution strings. The MISATTRIBUTION BAN is enforced by keeping these on the
# arm records themselves, so no consumer can render a baseline without one.
ATTRIBUTION = {
    "indra-default-source-prior": (
        "INDRA library default source priors (rand, syst) scored at one evidence. "
        "The 2023 paper's own belief arm refits these per fold by MCMC; this is the "
        "UNFITTED bundled default and is not that arm."
    ),
    "indra-bayes-source-oof": (
        "INDRA 1.24.0 BayesianScorer, per-source reliabilities refit out-of-fold from "
        "this panel's own evidence curation counts. The 2023 paper publishes NO "
        "Bayesian arm; this is library code, not a published method."
    ),
    "indra-bayes-subtype-oof": (
        "INDRA 1.24.0 BayesianScorer, per-(source, evidence subtype) reliabilities refit "
        "out-of-fold from this panel's own evidence curation counts. The 2023 paper "
        "publishes NO subtype arm; this is library code, not a published method."
    ),
}
READER_ATTRIBUTION = (
    "Our LLM evidence reader. NOT zero-shot: every call carries 14 hand-authored "
    "demonstration pairs. One verdict per (statement, evidence) pair; statement "
    "belief is these verdicts pushed through INDRA's noisy-OR."
)


class GateError(RuntimeError):
    """A contract this figure depends on did not hold."""


def fail(context: str, message: str):
    raise GateError(f"{context}: {message}")


# --------------------------------------------------------------------------
# io helpers
# --------------------------------------------------------------------------

def load_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, digest: bool = True) -> dict:
    return {
        "path": str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path) if digest else None,
    }


# --------------------------------------------------------------------------
# closed-form, tie-aware metrics over per-score-bucket contingency counts
# --------------------------------------------------------------------------

def auroc_from_buckets(pos: np.ndarray, neg: np.ndarray) -> float:
    """Tie-aware AUROC. `pos`/`neg` are counts per score bucket, ASCENDING score.

    Equals ``sklearn.roc_auc_score``: each positive is credited the negatives
    strictly below it plus half the negatives it ties with.
    """
    total_pos = pos.sum()
    total_neg = neg.sum()
    if total_pos == 0 or total_neg == 0:
        return float("nan")
    below = np.concatenate(([0.0], np.cumsum(neg[:-1].astype(np.float64))))
    return float((pos * (below + 0.5 * neg)).sum() / (total_pos * total_neg))


def average_precision_from_buckets(pos: np.ndarray, neg: np.ndarray) -> float:
    """Tie-aware average precision. Buckets are given ASCENDING; scanned DESCENDING.

    Equals ``sklearn.average_precision_score``: sum over distinct score
    thresholds of (recall increment x precision at that threshold). Tied scores
    form ONE threshold, which is exactly why the trapezoidal estimator is not
    used anywhere in this artifact.
    """
    total_pos = pos.sum()
    if total_pos == 0:
        return float("nan")
    tp = np.cumsum(pos[::-1].astype(np.float64))
    fp = np.cumsum(neg[::-1].astype(np.float64))
    seen = tp + fp
    keep = seen > 0
    precision = np.divide(tp, seen, out=np.zeros_like(tp), where=keep)
    recall = tp / total_pos
    increments = np.diff(np.concatenate(([0.0], recall)))
    return float((increments * precision).sum())


def error_f1_from_buckets(pos: np.ndarray, neg: np.ndarray, below: np.ndarray) -> dict:
    """Error-detection F1 at a fixed cut. Positive class = the evidence is INCORRECT.

    `below` is a boolean mask over ASCENDING buckets marking those predicted
    incorrect (score < threshold). `neg` counts are the true errors.
    """
    tp = float(neg[below].sum())
    fp = float(pos[below].sum())
    fn = float(neg[~below].sum())
    tn = float(pos[~below].sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    n = tp + fp + fn + tn
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": (tp + tn) / n if n else 0.0,
    }


def best_error_f1_from_buckets(pos: np.ndarray, neg: np.ndarray) -> float:
    """Best error-detection F1 over the arm's OWN score ladder.

    Reported alongside the fixed-cut number so a constant-within-source baseline
    is not strawmanned by a threshold chosen for the reader. Sweeps every cut
    point the arm can express, including "flag everything".
    """
    best = 0.0
    for cut in range(1, len(pos) + 1):
        mask = np.zeros(len(pos), dtype=bool)
        mask[:cut] = True
        best = max(best, error_f1_from_buckets(pos, neg, mask)["f1"])
    return best


# --------------------------------------------------------------------------
# arm score vectors -> bucket ids
# --------------------------------------------------------------------------

class Bucketed:
    """One arm's scores compressed to a small ascending ladder of distinct values.

    Every metric this script computes is a function of the (positives, negatives)
    contingency per distinct score, so an arm with 4 distinct scores costs 4
    numbers per bootstrap resample instead of 5,379.
    """

    def __init__(self, scores: np.ndarray):
        self.values, self.ids = np.unique(scores, return_inverse=True)
        self.n_buckets = int(self.values.size)
        self.below = self.values < DECISION_THRESHOLD

    def counts(self, labels: np.ndarray, take: np.ndarray | None = None):
        ids = self.ids if take is None else self.ids[take]
        y = labels if take is None else labels[take]
        flat = np.bincount(ids * 2 + y, minlength=self.n_buckets * 2)
        return flat[1::2], flat[0::2]  # positives, negatives


def point_metrics(bucketed: Bucketed, labels: np.ndarray, take: np.ndarray | None = None) -> dict:
    pos, neg = bucketed.counts(labels, take)
    # Error detection is the SAME curve with the classes swapped and the score
    # order reversed, so the buckets are simply read the other way round.
    err_pos, err_neg = neg[::-1], pos[::-1]
    return {
        "n": int(pos.sum() + neg.sum()),
        "n_positive": int(pos.sum()),
        "n_negative": int(neg.sum()),
        "auroc": auroc_from_buckets(pos, neg),
        "average_precision_correct": average_precision_from_buckets(pos, neg),
        "average_precision_incorrect": average_precision_from_buckets(err_pos, err_neg),
        "error_detection": error_f1_from_buckets(pos, neg, bucketed.below),
        "error_detection_best_f1": best_error_f1_from_buckets(pos, neg),
        "distinct_scores": bucketed.n_buckets,
    }


BOOTSTRAP_METRICS = ("auroc", "average_precision_correct", "average_precision_incorrect", "error_f1")


def bootstrap_row(bucketed: Bucketed, labels: np.ndarray, take: np.ndarray) -> dict:
    pos, neg = bucketed.counts(labels, take)
    if pos.sum() == 0 or neg.sum() == 0:
        return {k: float("nan") for k in BOOTSTRAP_METRICS}
    return {
        "auroc": auroc_from_buckets(pos, neg),
        "average_precision_correct": average_precision_from_buckets(pos, neg),
        "average_precision_incorrect": average_precision_from_buckets(neg[::-1], pos[::-1]),
        "error_f1": error_f1_from_buckets(pos, neg, bucketed.below)["f1"],
    }


def interval(values: list[float]) -> dict:
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n_valid_resamples": 0}
    return {
        "mean": float(arr.mean()),
        "ci95_low": float(np.percentile(arr, 2.5)),
        "ci95_high": float(np.percentile(arr, 97.5)),
        "n_valid_resamples": int(arr.size),
    }


def delta_interval(values: list[float]) -> dict:
    out = interval(values)
    if out["n_valid_resamples"] == 0:
        out["excludes_zero"] = False
        out["p_arm_greater"] = None
        return out
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=np.float64)
    out["excludes_zero"] = bool(out["ci95_low"] > 0 or out["ci95_high"] < 0)
    out["p_arm_greater"] = float((arr > 0).mean())
    return out


# --------------------------------------------------------------------------
# reader verdict recovery
# --------------------------------------------------------------------------

# Mirrors `statement_belief._dedup_priority`'s readable-beats-unreadable rule for
# the narrow case that occurs in these files: a retry pair where one row parsed
# and the other did not. Any genuine two-verdict conflict on one key is a gate.
def recover_reader_rows(path: Path, arm_id: str) -> tuple[dict, dict, str]:
    """Stream one raw attempts file -> {(stmt_hash, source_hash): row}, census, sha256."""
    digest = hashlib.sha256()
    by_key: dict[tuple[str, str], dict] = {}
    census = Counter()
    conflicts: list[str] = []
    with path.open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            if not raw.strip():
                continue
            record = json.loads(raw)
            census["raw_rows"] += 1
            if record.get("row_status") != "scored":
                census[f"skipped_row_status_{record.get('row_status')}"] += 1
                continue
            verdict = record.get("verdict")
            if verdict is None:
                census["skipped_null_verdict"] += 1
                continue
            key = (str(record["paper_statement_hash"]), str(record["source_hash"]))
            row = {
                "source_api": record.get("source_api"),
                "verdict": verdict,
                "score": record.get("score"),
                "confidence": record.get("confidence"),
                "tier": record.get("tier"),
                "grounding_status": record.get("grounding_status"),
            }
            previous = by_key.get(key)
            if previous is not None:
                if previous["verdict"] != verdict or previous["score"] != row["score"]:
                    conflicts.append(f"{key}: {previous} vs {row}")
                census["duplicate_scored_rows_reconciled"] += 1
                continue
            by_key[key] = row
    if conflicts:
        fail(
            f"reader[{arm_id}]",
            f"{len(conflicts)} scored duplicate key(s) disagree on the verdict, "
            f"so no per-evidence measurement is well defined: {conflicts[:3]}",
        )
    return by_key, dict(census), digest.hexdigest()


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------

def simple_prior_probability(rand: float, syst: float) -> float:
    """INDRA's own single-evidence belief.

    ``SimpleScorer.score_evidence_list`` on one evidence from one source reduces
    to ``1 - (syst_s + rand_s)`` -- the per-source pair is the whole model, so
    this quantity is CONSTANT within a source by construction.
    """
    return 1.0 - (syst + rand)


def load_bayes_folds(path: Path, arm_id: str) -> dict[int, dict]:
    rows = [r for r in load_jsonl(path) if r["arm_id"] == arm_id]
    if len(rows) != 10:
        fail(f"bayes[{arm_id}]", f"expected 10 out-of-fold rows, got {len(rows)}")
    by_fold = {int(r["fit_fold_id"]): r for r in rows}
    if set(by_fold) != set(range(10)):
        fail(f"bayes[{arm_id}]", "fit_fold_id coverage is not 0..9")
    return by_fold


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in DEFAULTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", default=default)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--skip-reaggregation",
        action="store_true",
        help="Skip the statement-grain reconciliation gate. Development only: the "
             "shipped artifact records reaggregation.verified=false and the viewer "
             "gates the figure on it.",
    )
    args = parser.parse_args()

    paths = {name: (REPO / getattr(args, name)) for name in DEFAULTS}
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- gold: the per-evidence adjudication IS the panel --------------------
    adjudication = list(load_jsonl(paths["evidence_gold"]))
    gold: dict[tuple[str, str], int] = {}
    review_status = Counter()
    curators = Counter()
    for record in adjudication:
        key = (str(record["paper_statement_hash"]), str(record["source_hash"]))
        if record["identity_kind"] != "statement_source_hash_pair":
            fail("gold", f"{key}: unexpected identity_kind {record['identity_kind']!r}")
        review_status[record["review_status"]] += 1
        if record["review_status"] == "unreviewed":
            continue
        label = record["evidence_gold_label"]
        if label not in (0, 1):
            fail("gold", f"{key}: reviewed pair carries label {label!r}")
        if key in gold:
            fail("gold", f"{key}: duplicate reviewed pair")
        gold[key] = label
        for curation in record["curations"]:
            curators[curation["curator"]] += 1

    # Independent cross-check against the statement gold's own evidence_review
    # blocks. Two files, one truth: a drift in either gates the run.
    statement_gold = list(load_jsonl(paths["statement_gold"]))
    cross: dict[tuple[str, str], int] = {}
    for record in statement_gold:
        stmt = str(record["paper_statement_hash"])
        review = record["evidence_review"]
        for source_hash in review["positive_source_hashes"]:
            cross[(stmt, str(source_hash))] = 1
        for source_hash in review["negative_source_hashes"]:
            cross[(stmt, str(source_hash))] = 0
    if cross != gold:
        fail(
            "gold",
            f"paper_evidence_adjudication ({len(gold)} reviewed pairs) disagrees with "
            f"paper_statement_gold.evidence_review ({len(cross)})",
        )

    # ---- execution map: one row per evidence read ---------------------------
    execution = {}
    for record in load_jsonl(paths["execution_map"]):
        key = (str(record["paper_statement_hash"]), str(record["source_hash"]))
        if key in execution:
            fail("execution_map", f"{key}: duplicate unique-pair row")
        execution[key] = record
    missing = [k for k in gold if k not in execution]
    if missing:
        fail("execution_map", f"{len(missing)} reviewed pair(s) were never executed: {missing[:3]}")

    # ---- folds --------------------------------------------------------------
    fold_by_statement_id = {r["statement_id"]: int(r["fold_id"]) for r in load_jsonl(paths["fold_gold"])}
    statement_id_by_hash = {
        str(r["paper_statement_hash"]): r["canonical_corpus"]["statement_id"] for r in statement_gold
    }
    fold_by_hash = {h: fold_by_statement_id[s] for h, s in statement_id_by_hash.items()}

    # ---- ordered panel ------------------------------------------------------
    # Deterministic key order fixes the bootstrap index semantics for every arm.
    keys = sorted(gold)
    labels = np.array([gold[k] for k in keys], dtype=np.int64)
    sources = [execution[k]["source_api"] for k in keys]
    folds = np.array([fold_by_hash[k[0]] for k in keys], dtype=np.int64)
    n_pairs = len(keys)

    # ---- readers ------------------------------------------------------------
    arms: list[dict] = []
    reader_rows: dict[str, dict] = {}
    reader_inputs: dict[str, dict] = {}
    coverage_by_arm: dict[str, dict] = {}
    for arm_id, arm_dir, display in READER_ARMS:
        manifest_path = REPO / args.models_dir / arm_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        declared = manifest["implementation"]["notes"]["inputs"].get("raw_attempts")
        if declared is None:
            fail(f"reader[{arm_id}]", "bundle manifest declares no raw_attempts input")
        raw_path = (manifest_path.parent / declared["path"]).resolve()
        if not raw_path.is_file():
            fail(f"reader[{arm_id}]", f"declared raw_attempts {raw_path} is missing")
        rows, census, digest = recover_reader_rows(raw_path, arm_id)
        if digest != declared["sha256"]:
            fail(
                f"reader[{arm_id}]",
                f"raw_attempts sha256 {digest} != manifest-declared {declared['sha256']}",
            )
        unscored = [k for k in gold if k not in rows]
        coverage_by_arm[arm_id] = {
            "raw_attempts_path": str(raw_path.relative_to(REPO)),
            "raw_attempts_sha256": digest,
            "raw_attempts_sha256_matches_manifest": True,
            "scored_unique_pairs": len(rows),
            "execution_map_pairs": len(execution),
            "reviewed_pairs_scored": len(gold) - len(unscored),
            "reviewed_pairs_unscored": len(unscored),
            "unscored_keys": [list(k) for k in unscored[:20]],
            "row_census": census,
            "tier_census_reviewed": dict(Counter(rows[k]["tier"] for k in gold if k in rows)),
        }
        if unscored:
            # Visible, never silently excluded. A reviewed pair with no reader
            # measurement is a hole in the comparison, so it gates.
            fail(
                f"reader[{arm_id}]",
                f"{len(unscored)} reviewed pair(s) carry no reader verdict; the "
                f"per-evidence panel would silently differ from the other arms",
            )
        reader_rows[arm_id] = rows
        reader_inputs[arm_id] = {"manifest": manifest, "manifest_path": manifest_path}
        scores = np.array([rows[k]["score"] for k in keys], dtype=np.float64)
        if not np.isfinite(scores).all():
            fail(f"reader[{arm_id}]", "non-finite per-evidence score")
        arms.append({
            "id": arm_id,
            "display": display,
            "kind": "reader",
            "attribution": READER_ATTRIBUTION,
            "join_key": arm_dir,
            "provider_model_id": manifest["implementation"]["notes"].get("provider_model_id"),
            "constant_within_source": False,
            "scores": scores,
        })

    # ---- baseline 1: INDRA bundled per-source prior --------------------------
    aggregation = json.loads(paths["aggregation"].read_text())
    priors = {k: (float(v[0]), float(v[1])) for k, v in aggregation["priors"].items()}
    bundled = json.loads(paths["default_belief_probs"].read_text())
    for source, (rand, syst) in priors.items():
        if source not in bundled["rand"] or source not in bundled["syst"]:
            continue
        if abs(bundled["rand"][source] - rand) > 1e-12 or abs(bundled["syst"][source] - syst) > 1e-12:
            fail("priors", f"{source}: aggregation.json disagrees with default_belief_probs.json")
    default_scores = np.array(
        [simple_prior_probability(*priors[src]) for src in sources], dtype=np.float64
    )
    arms.append({
        "id": "indra-default-source-prior",
        "display": "INDRA source prior (bundled)",
        "kind": "baseline",
        "attribution": ATTRIBUTION["indra-default-source-prior"],
        "join_key": "indra-default-source-prior",
        "provider_model_id": None,
        "constant_within_source": True,
        "scores": default_scores,
    })

    # ---- baselines 2/3: BayesianScorer, refit out-of-fold --------------------
    subtype_by_key, subtype_note = tag_subtypes(paths["corpus"], statement_gold, execution)
    for arm_id, provenance_arm, display, use_subtype in (
        ("indra-bayes-source-oof", BAYES_SOURCE_ARM, "INDRA Bayes source (OOF)", False),
        ("indra-bayes-subtype-oof", BAYES_SUBTYPE_ARM, "INDRA Bayes subtype (OOF)", True),
    ):
        if use_subtype and subtype_by_key is None:
            arms.append({
                "id": arm_id, "display": display, "kind": "baseline",
                "attribution": ATTRIBUTION[arm_id], "join_key": arm_id,
                "provider_model_id": None, "constant_within_source": True,
                "scores": None, "unavailable_reason": subtype_note,
            })
            continue
        by_fold = load_bayes_folds(paths["bayes_provenance"], provenance_arm)
        values = np.empty(n_pairs, dtype=np.float64)
        path_census = Counter()
        for i, key in enumerate(keys):
            fold = by_fold[int(folds[i])]
            source = sources[i]
            fitted = fold["fitted_source_parameters"].get(source)
            subtype_params = fold.get("fitted_subtype_parameters", {}).get(source, {})
            subtype = subtype_by_key.get(key) if use_subtype else None
            if use_subtype and subtype is not None and subtype in subtype_params:
                rand = float(subtype_params[subtype]["random_error"])
                syst = float(fitted["systematic_error"]) if fitted else priors[source][1]
                path_census["fitted_concrete_subtype"] += 1
            elif fitted is not None:
                rand = float(fitted["random_error"])
                syst = float(fitted["systematic_error"])
                path_census["fitted_source"] += 1
            else:
                rand, syst = priors[source]
                path_census["bundled_default_source"] += 1
            values[i] = simple_prior_probability(rand, syst)
        arms.append({
            "id": arm_id, "display": display, "kind": "baseline",
            "attribution": ATTRIBUTION[arm_id], "join_key": arm_id,
            "provider_model_id": None, "constant_within_source": not use_subtype,
            "scores": values, "parameter_path_census": dict(path_census),
        })

    drawable = [a for a in arms if a.get("scores") is not None]

    # ---- the estimator agreement gate ---------------------------------------
    sklearn_check = {}
    for arm in drawable:
        bucketed = Bucketed(arm["scores"])
        pos, neg = bucketed.counts(labels)
        fast_auroc = auroc_from_buckets(pos, neg)
        fast_ap = average_precision_from_buckets(pos, neg)
        ref_auroc = float(roc_auc_score(labels, arm["scores"]))
        ref_ap = float(average_precision_score(labels, arm["scores"]))
        if abs(fast_auroc - ref_auroc) > SKLEARN_TOLERANCE or abs(fast_ap - ref_ap) > SKLEARN_TOLERANCE:
            fail(
                f"estimator[{arm['id']}]",
                f"closed form disagrees with sklearn: auroc {fast_auroc} vs {ref_auroc}, "
                f"ap {fast_ap} vs {ref_ap}",
            )
        sklearn_check[arm["id"]] = {
            "auroc_abs_diff": abs(fast_auroc - ref_auroc),
            "average_precision_abs_diff": abs(fast_ap - ref_ap),
        }
        arm["_bucketed"] = bucketed

    # ---- the grain bridge: verdicts must rebuild the shipped statement number -
    reaggregation = {"verified": not args.skip_reaggregation, "arms": {}}
    if not args.skip_reaggregation:
        reaggregation["arms"] = reaggregate(
            paths["corpus"], statement_gold, execution, reader_rows, priors,
            REPO / args.models_dir, {a: d for a, d, _ in READER_ARMS},
        )
        worst = max((v["max_abs_diff"] for v in reaggregation["arms"].values()), default=0.0)
        if worst > REAGGREGATION_TOLERANCE:
            fail(
                "reaggregation",
                f"recovered per-evidence verdicts do not rebuild the shipped statement "
                f"probabilities (max |Δ| = {worst}); the two grains are not the same run",
            )
        reaggregation["max_abs_diff"] = worst

    # ---- census -------------------------------------------------------------
    source_census = []
    for source, total in Counter(sources).most_common():
        idx = [i for i, s in enumerate(sources) if s == source]
        positives = int(labels[idx].sum())
        source_census.append({
            "source": source,
            "reviewed_pairs": total,
            "positive_pairs": positives,
            "negative_pairs": total - positives,
            "observed_correct_fraction": positives / total,
            "executed_pairs": sum(1 for r in execution.values() if r["source_api"] == source),
            "bundled_prior_at_one_evidence": simple_prior_probability(*priors[source]),
            "metric_row": total >= MIN_STRATUM_N and 0 < positives < total,
        })

    strata = [row["source"] for row in source_census if row["metric_row"]]
    shared_prior = shared_prior_defect(source_census, priors)

    # ---- point metrics, pooled and per source -------------------------------
    for arm in drawable:
        arm["metrics"] = point_metrics(arm["_bucketed"], labels)
        per_source = {}
        for source in strata:
            take = np.array([i for i, s in enumerate(sources) if s == source], dtype=np.int64)
            per_source[source] = point_metrics(arm["_bucketed"], labels, take)
        arm["per_source"] = per_source

    # ---- source-stratified paired bootstrap ---------------------------------
    rng = np.random.default_rng(args.seed)
    index_by_source = {s: np.array([i for i, x in enumerate(sources) if x == s], dtype=np.int64)
                       for s in sorted(set(sources))}
    stratum_order = sorted(index_by_source)
    stratum_set = set(strata)
    pooled_draws = {a["id"]: {m: [] for m in BOOTSTRAP_METRICS} for a in drawable}
    source_draws = {a["id"]: {s: {m: [] for m in BOOTSTRAP_METRICS} for s in strata}
                    for a in drawable}
    for _ in range(args.n_boot):
        # ONE shared index vector per resample, consumed by every arm and by every
        # stratum, so all deltas below are genuinely paired.
        take = np.concatenate([
            rng.choice(index_by_source[s], size=index_by_source[s].size, replace=True)
            for s in stratum_order
        ])
        take_by_source = {}
        offset = 0
        for s in stratum_order:
            size = index_by_source[s].size
            if s in stratum_set:
                take_by_source[s] = take[offset:offset + size]
            offset += size
        for arm in drawable:
            row = bootstrap_row(arm["_bucketed"], labels, take)
            for metric in BOOTSTRAP_METRICS:
                pooled_draws[arm["id"]][metric].append(row[metric])
            for source, sub in take_by_source.items():
                sub_row = bootstrap_row(arm["_bucketed"], labels, sub)
                for metric in BOOTSTRAP_METRICS:
                    source_draws[arm["id"]][source][metric].append(sub_row[metric])

    for arm in drawable:
        arm["metrics"]["interval"] = {
            metric: interval(pooled_draws[arm["id"]][metric]) for metric in BOOTSTRAP_METRICS
        }
        for source in strata:
            arm["per_source"][source]["interval"] = {
                metric: interval(source_draws[arm["id"]][source][metric])
                for metric in BOOTSTRAP_METRICS
            }

    # ---- paired deltas vs the strongest sourceable baseline ------------------
    baselines = [a for a in drawable if a["kind"] == "baseline"]
    reference = max(baselines, key=lambda a: a["metrics"]["auroc"])
    deltas = {}
    for arm in drawable:
        if arm["id"] == reference["id"]:
            continue
        deltas[arm["id"]] = {
            "pooled": {
                metric: delta_interval([
                    a - b for a, b in zip(pooled_draws[arm["id"]][metric],
                                          pooled_draws[reference["id"]][metric])
                ])
                for metric in BOOTSTRAP_METRICS
            },
            "point": {
                "auroc": arm["metrics"]["auroc"] - reference["metrics"]["auroc"],
                "average_precision_correct": (
                    arm["metrics"]["average_precision_correct"]
                    - reference["metrics"]["average_precision_correct"]
                ),
                "average_precision_incorrect": (
                    arm["metrics"]["average_precision_incorrect"]
                    - reference["metrics"]["average_precision_incorrect"]
                ),
                "error_f1": (
                    arm["metrics"]["error_detection"]["f1"]
                    - reference["metrics"]["error_detection"]["f1"]
                ),
            },
            "per_source": {
                source: {
                    metric: delta_interval([
                        a - b for a, b in zip(source_draws[arm["id"]][source][metric],
                                              source_draws[reference["id"]][source][metric])
                    ])
                    for metric in BOOTSTRAP_METRICS
                }
                for source in strata
            },
        }

    # ---- the other half of the bridge: the SAME model at statement grain -----
    statement_grain = statement_grain_metrics(statement_gold, paths["statement_headtohead"])
    for arm in drawable:
        block = statement_grain["arms"].get(arm["id"])
        if block is None:
            fail(f"statement_grain[{arm['id']}]", "no statement-grain prediction file is paired")
        arm["statement_grain"] = block

    # ---- contamination, at source_hash grain --------------------------------
    contamination = contamination_check(paths["prompt_manifest"], paths["corpus"],
                                        statement_gold, execution, gold)
    # Sensitivity, not exclusion: the overlapping pairs stay in the primary panel
    # (dropping items after seeing them is its own bias) and the figure carries
    # what the numbers would have been without them.
    leaked = {tuple(k) for k in contamination["overlapping_pairs"]}
    if leaked:
        keep = np.array([i for i, k in enumerate(keys) if k not in leaked], dtype=np.int64)
        clean = {}
        for arm in drawable:
            metrics = point_metrics(arm["_bucketed"], labels, keep)
            clean[arm["id"]] = {
                "auroc": metrics["auroc"],
                "average_precision_correct": metrics["average_precision_correct"],
                "average_precision_incorrect": metrics["average_precision_incorrect"],
                "error_f1": metrics["error_detection"]["f1"],
                "auroc_abs_shift": abs(metrics["auroc"] - arm["metrics"]["auroc"]),
            }
        contamination["sensitivity"] = {
            "n_pairs_excluded": int(n_pairs - keep.size),
            "n_pairs_kept": int(keep.size),
            "policy": "reported as a sensitivity; the primary panel keeps every reviewed pair",
            "arms": clean,
            "max_auroc_abs_shift": max(v["auroc_abs_shift"] for v in clean.values()),
        }

    # ---- power comparison against the statement grain ------------------------
    power = {
        "n_evidence_pairs": n_pairs,
        "n_statements": len(statement_gold),
        "ratio": n_pairs / len(statement_gold),
        "note": (
            "The statement panel is 1,689 items; this one is 5,379 reviewed evidence "
            "pairs from the same statements. Pairs within a statement are not "
            "independent, so the effective gain is below the raw 3.2x ratio; the "
            "bootstrap resamples pairs within source and does not model that "
            "clustering."
        ),
    }

    # ---- write --------------------------------------------------------------
    artifact = {
        "artifact_kind": ARTIFACT_KIND,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "n_bootstrap": args.n_boot,
        "decision_threshold": DECISION_THRESHOLD,
        "bootstrap_design": (
            "source-stratified paired bootstrap; one shared index vector per resample "
            "across every arm and every stratum"
        ),
        "estimator_contract": {
            "auroc": "tie-aware, equals sklearn.metrics.roc_auc_score",
            "average_precision": "tie-aware, equals sklearn.metrics.average_precision_score",
            "trapezoidal_pr_auc": (
                "deliberately absent: the reader's per-evidence score takes at most "
                "five distinct values, the regime where trapezoidal interpolation "
                "inflates most"
            ),
            "error_detection_positive_class": "the evidence pair is INCORRECT",
            "sklearn_agreement": sklearn_check,
            "sklearn_version": __import__("sklearn").__version__,
        },
        "panel": {
            "id": PANEL_ID,
            "n_reviewed_pairs": n_pairs,
            "n_positive": int(labels.sum()),
            "n_negative": int(n_pairs - labels.sum()),
            "negative_fraction": float(1 - labels.mean()),
            "n_statements": len(statement_gold),
            "n_statements_with_reviewed_pair": len({k[0] for k in keys}),
            "curators": [{"curator": c, "reviewed_pairs": n} for c, n in curators.most_common()],
        },
        "coverage": {
            "executed_unique_pairs": len(execution),
            "review_status_census": dict(review_status),
            "reviewed_pairs": n_pairs,
            "unreviewed_pairs": review_status["unreviewed"],
            "per_arm": coverage_by_arm,
            "sources": source_census,
            "min_stratum_n_for_metric_row": MIN_STRATUM_N,
            "excluded_baselines": [{
                "family": "paper Table 6 supervised rows (RF / Log LR / KNN / SVC, "
                          "with and without Type/#PMIDs/promoter/avglen)",
                "reason": (
                    "statement-grain only: every feature is a statement-level aggregate "
                    "(#PMIDs, statement type, promoter presence, mean evidence length) "
                    "and the models emit no per-evidence quantity, so no per-evidence "
                    "score exists to draw"
                ),
            }],
        },
        "reaggregation": reaggregation,
        "statement_grain": {k: v for k, v in statement_grain.items() if k != "arms"},
        "power": power,
        "shared_prior_defect": shared_prior,
        "contamination": contamination,
        "reference_arm_id": reference["id"],
        "strata": strata,
        "subtype_policy": subtype_note,
        "arms": [
            {k: v for k, v in arm.items() if k not in ("scores", "_bucketed")}
            for arm in arms
        ],
        "paired_delta_vs_reference": deltas,
        "inputs": {name: file_record(path) for name, path in paths.items() if path.is_file()},
    }

    out_json = out_dir / "per_evidence_comparison.json"
    out_json.write_text(json.dumps(artifact, indent=1, sort_keys=True, default=_json_default) + "\n")

    pairs_path = out_dir / "per_evidence_pairs.jsonl"
    with pairs_path.open("w") as handle:
        for i, key in enumerate(keys):
            row = {
                "paper_statement_hash": key[0],
                "source_hash": key[1],
                "source_api": sources[i],
                "fold_id": int(folds[i]),
                "gold_label": int(labels[i]),
                "statement_type": execution[key]["statement_type"],
                "route": execution[key]["route"],
            }
            for arm in drawable:
                row[arm["id"]] = float(arm["scores"][i])
            for arm_id, rows in reader_rows.items():
                row[f"{arm_id}__verdict"] = rows[key]["verdict"]
                row[f"{arm_id}__tier"] = rows[key]["tier"]
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    markdown = render_markdown(artifact)
    (out_dir / "per_evidence_comparison.md").write_text(markdown)
    print(markdown)
    print(f"\nwrote {out_json}\nwrote {pairs_path}")
    return 0


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"not JSON serialisable: {type(value)}")


# --------------------------------------------------------------------------
# subtype tagging (INDRA 1.24.0), optional
# --------------------------------------------------------------------------

def tag_subtypes(corpus_path: Path, statement_gold: list, execution: dict):
    """(statement, source_hash) -> INDRA evidence subtype, for the read that happened.

    The unique-pairs corpus is already de-duplicated to exactly one evidence entry
    per ``(statement, source_hash)`` -- 33,361 entries for 33,361 pairs -- so there
    is no choice to make and none is faked; the uniqueness is asserted rather than
    assumed. (``pair_multiplicity`` in the execution map records how many raw
    evidence entries collapsed into each pair; that matters for re-aggregation,
    not for subtype tagging, which is a property of the sentence.)

    Returns (None, reason) when INDRA is unavailable, so the subtype baseline
    gates itself out instead of taking the whole run down.
    """
    try:
        from indra.belief import tag_evidence_subtype
        from indra.statements import Evidence
    except Exception as error:  # pragma: no cover - environment dependent
        return None, f"INDRA is not importable, so no evidence subtype can be tagged: {error}"

    corpus = json.loads(corpus_path.read_text())
    if len(corpus) != len(statement_gold):
        return None, "corpus and statement gold disagree on statement count"
    by_key: dict[tuple[str, str], str | None] = {}
    for record, statement in zip(statement_gold, corpus):
        stmt = str(record["paper_statement_hash"])
        for evidence in statement["evidence"]:
            key = (stmt, str(evidence["source_hash"]))
            if key not in execution:
                continue
            if key in by_key:
                fail(
                    "subtype",
                    f"{key}: the unique-pairs corpus carries more than one evidence "
                    f"entry for one pair, so 'the sentence that was read' is ambiguous",
                )
            _, subtype = tag_evidence_subtype(Evidence._from_json(evidence))
            by_key[key] = subtype
    return by_key, (
        "subtype tagged with indra.belief.tag_evidence_subtype on the one corpus "
        "evidence entry each (statement, source_hash) pair has; a null or unseen "
        "subtype falls back to the fold's fitted source parameter"
    )


# --------------------------------------------------------------------------
# the grain bridge
# --------------------------------------------------------------------------

def reaggregate(corpus_path: Path, statement_gold: list, execution: dict,
                reader_rows: dict, priors: dict, models_dir: Path,
                arm_dir_by_id: dict[str, str]) -> dict:
    """Rebuild each arm's SHIPPED statement probability from the recovered verdicts.

    This is the whole premise of the figure: if the per-evidence rows are the ones
    behind the statement numbers, pushing them through ``statement_belief`` with
    the run's own priors must land on ``all_source_predictions.jsonl`` exactly.

    The corpus is de-duplicated to unique pairs, so each measured pair is expanded
    back by its ``pair_multiplicity`` before aggregation: 635 pairs stand for more
    than one raw evidence entry, and INDRA's noisy-OR counts raw entries.
    """
    corpus = json.loads(corpus_path.read_text())
    texts: dict[tuple[str, str], list] = defaultdict(list)
    for record, statement in zip(statement_gold, corpus):
        stmt = str(record["paper_statement_hash"])
        for evidence in statement["evidence"]:
            texts[(stmt, str(evidence["source_hash"]))].append(evidence.get("text"))

    by_statement: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in execution:
        by_statement[key[0]].append(key)

    out = {}
    for arm_id, rows in reader_rows.items():
        predictions_path = models_dir / arm_dir_by_id[arm_id] / "all_source_predictions.jsonl"
        shipped = {r["statement_id"]: r["probability_correct"] for r in load_jsonl(predictions_path)}
        worst = 0.0
        exact = 0
        for record in statement_gold:
            stmt = str(record["paper_statement_hash"])
            statement_id = record["canonical_corpus"]["statement_id"]
            evidence_rows = []
            for key in by_statement[stmt]:
                row = rows.get(key)
                if row is None:
                    continue
                available = texts.get(key, [None])
                for i in range(execution[key]["pair_multiplicity"]):
                    evidence_rows.append({
                        "source_api": row["source_api"],
                        "verdict": row["verdict"],
                        "confidence": row["confidence"],
                        "tier": row["tier"],
                        "evidence_text": available[i] if i < len(available) else available[0],
                        "evidence_hash": None,
                    })
            rolled = statement_belief(evidence_rows, priors)
            value = rolled.belief if rolled.belief is not None else 0.0
            difference = abs(value - shipped[statement_id])
            worst = max(worst, difference)
            exact += int(difference == 0.0)
        out[arm_id] = {
            "n_statements": len(statement_gold),
            "n_exact": exact,
            "max_abs_diff": worst,
            "shipped_predictions": str(predictions_path.relative_to(REPO)),
            "aggregation": "indra_default_hard_gate",
        }
    return out


# --------------------------------------------------------------------------
# the shared-prior defect, measured at per-evidence grain on THIS panel
# --------------------------------------------------------------------------

def shared_prior_defect(source_census: list, priors: dict) -> dict:
    """INDRA's default prior assigns one number to sources that behave differently.

    Chi-square on the reviewed pairs' correct/incorrect counts across the sources
    that SHARE a single prior value, so the test is about the sources the prior
    actually conflates rather than all sources at once.
    """
    groups = defaultdict(list)
    for row in source_census:
        if row["reviewed_pairs"] < MIN_STRATUM_N:
            continue
        groups[round(row["bundled_prior_at_one_evidence"], 12)].append(row)
    blocks = []
    for value, rows in sorted(groups.items(), reverse=True):
        if len(rows) < 2:
            continue
        observed = np.array([[r["positive_pairs"], r["negative_pairs"]] for r in rows], dtype=np.float64)
        total = observed.sum()
        expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total
        chi2 = float(((observed - expected) ** 2 / expected).sum())
        dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
        blocks.append({
            "shared_prior_at_one_evidence": value,
            "sources": [{
                "source": r["source"],
                "reviewed_pairs": r["reviewed_pairs"],
                "observed_correct_fraction": r["observed_correct_fraction"],
            } for r in sorted(rows, key=lambda r: r["observed_correct_fraction"])],
            "observed_correct_fraction_min": min(r["observed_correct_fraction"] for r in rows),
            "observed_correct_fraction_max": max(r["observed_correct_fraction"] for r in rows),
            "chi2": chi2,
            "dof": dof,
            "p_value": _chi2_sf(chi2, dof),
        })
    return {
        "blocks": blocks,
        "note": (
            "Measured on THIS panel's reviewed evidence pairs. Other panels on this "
            "site measure the same defect on different gold and report a different "
            "spread; the defect is the shared prior, not any one spread."
        ),
    }


def _chi2_sf(chi2: float, dof: int) -> float:
    """Upper tail of the chi-square distribution, via the regularised gamma Q.

    Kept local so this artifact does not take a scipy dependency for one number.
    """
    if dof <= 0 or chi2 < 0:
        return float("nan")
    a, x = dof / 2.0, chi2 / 2.0
    if x == 0:
        return 1.0
    if x < a + 1:
        # series for P(a, x)
        term = 1.0 / a
        total = term
        for n in range(1, 1000):
            term *= x / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return 1.0 - total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    # continued fraction for Q(a, x)
    tiny = 1e-300
    b, c, d = x + 1 - a, 1 / tiny, 1 / (x + 1 - a)
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-16:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


# --------------------------------------------------------------------------
# contamination, at source_hash grain
# --------------------------------------------------------------------------

def _statement_agents(statement: dict) -> list[str]:
    """Every agent name on an INDRA statement, whatever its type.

    ``members`` is Complex-only. Binary statements carry their agents on
    ``subj``/``obj``/``enz``/``sub``/``agent``, and a checker that reads only
    ``members`` sees NOTHING but Complex statements while still looking like it
    passed. That is the vacuity mode this function exists to close.
    """
    names: list[str] = []
    for member in statement.get("members") or []:
        name = member.get("name") or (member.get("db_refs") or {}).get("TEXT")
        if name:
            names.append(str(name))
    for field in ("subj", "obj", "enz", "sub", "agent", "agent_from", "agent_to"):
        agent = statement.get(field)
        if isinstance(agent, dict):
            name = agent.get("name") or (agent.get("db_refs") or {}).get("TEXT")
            if name:
                names.append(str(name))
    return names


def _parse_demo_claim(claim: str) -> tuple[list[str], str] | None:
    """``A + B [Complex]`` or ``A [Type] B`` -> (agent names, type).

    The Complex form is tried FIRST. Matching the binary pattern first swallows
    "MYB + PPID [Complex]" into the single agent name "MYB + PPID", which can
    never equal a two-element panel agent set — a silent false negative.
    """
    import re

    complex_form = re.match(r"^(.*?)\s*\+\s*(.*?)\s*\[(\w+)\]\s*$", claim)
    if complex_form:
        return [complex_form.group(1).strip(), complex_form.group(2).strip()], complex_form.group(3)
    binary = re.match(r"^(.*?)\s*\[(\w+)\]\s*(.*)$", claim)
    if binary:
        agents = [x.strip() for x in (binary.group(1), binary.group(3)) if x.strip()]
        return agents, binary.group(2)
    return None


def contamination_check(prompt_manifest: Path, corpus_path: Path,
                        statement_gold: list, execution: dict, gold: dict) -> dict:
    """Do any demonstration EVIDENCE sentences appear among the scored pairs?

    ``tests/test_paper_panel_fewshot_disjoint.py`` checks (agent set, statement
    type) grain. This panel's unit is the evidence PAIR, so the check is redone at
    that grain: the normalised demonstration sentence against the normalised text
    of every reviewed pair's corpus evidence. Each hit is reported with enough
    context to judge it -- the panel statement's own agents and type, the
    demonstration verdict, and whether that verdict agrees with the curators.
    """
    import re

    manifest = json.loads(prompt_manifest.read_text())
    demos: list[dict] = []
    for index, prefix in enumerate(manifest["prompt_components"]["main_message_prefixes"]):
        messages = prefix["messages"]
        for position, message in enumerate(messages):
            if message.get("role") != "user":
                continue
            claim = re.search(r"CLAIM:\s*(.+)", message["content"])
            evidence = re.search(r"EVIDENCE:\s*(.+)", message["content"])
            if not (claim and evidence):
                continue
            verdict = None
            if position + 1 < len(messages) and messages[position + 1].get("role") == "assistant":
                try:
                    verdict = json.loads(messages[position + 1]["content"]).get("verdict")
                except (ValueError, AttributeError):
                    verdict = None
            demos.append({
                "prefix": index,
                "claim": claim.group(1).strip(),
                "sentence": _normalise(evidence.group(1)),
                "verdict": verdict,
            })
    if not demos:
        fail("contamination", "parsed no demonstration pairs — the check would be vacuous")
    by_sentence: dict[str, list[dict]] = defaultdict(list)
    for demo in demos:
        by_sentence[demo["sentence"]].append(demo)

    corpus = json.loads(corpus_path.read_text())
    reviewed_texts: dict[str, list] = defaultdict(list)
    hits: list[dict] = []
    scanned = 0
    for record, statement in zip(statement_gold, corpus):
        stmt = str(record["paper_statement_hash"])
        for evidence in statement["evidence"]:
            key = (stmt, str(evidence["source_hash"]))
            if key not in gold:
                continue
            text = evidence.get("text")
            if not text:
                continue
            scanned += 1
            normalised = _normalise(text)
            reviewed_texts[normalised].append(key)
            matches = by_sentence.get(normalised)
            if not matches:
                continue
            panel_agents = frozenset(_statement_agents(statement))
            panel_type = str(statement.get("type", "")).lower()
            label = gold[key]
            same_claim = False
            for demo in matches:
                parsed = _parse_demo_claim(demo["claim"])
                if parsed and frozenset(parsed[0]) == panel_agents and parsed[1].lower() == panel_type:
                    same_claim = True
            hits.append({
                "paper_statement_hash": key[0],
                "source_hash": key[1],
                "source_api": execution[key]["source_api"],
                "gold_label": label,
                "panel_statement_type": statement.get("type"),
                "panel_agents": sorted(panel_agents),
                "demonstration_claims": sorted({d["claim"] for d in matches}),
                "demonstration_verdicts": sorted({str(d["verdict"]) for d in matches}),
                "demonstration_prefixes": sorted({d["prefix"] for d in matches}),
                # True = the demonstration shows the model the SAME claim on the
                # SAME sentence; the answer itself, not just the sentence, leaked.
                "same_agent_set_and_type": same_claim,
                "demonstration_verdict_agrees_with_gold": sorted({
                    (d["verdict"] == "correct") == bool(label) for d in matches
                }),
            })

    # Non-vacuity: the two sides must share vocabulary, or a null intersection
    # proves nothing. Token overlap is the cheapest honest witness.
    demo_tokens: set[str] = set()
    for sentence in by_sentence:
        demo_tokens.update(sentence.split())
    panel_tokens: set[str] = set()
    for text in reviewed_texts:
        panel_tokens.update(text.split())
    shared = demo_tokens & panel_tokens
    if not shared:
        fail("contamination", "demonstration and panel vocabularies do not intersect at all")

    same_claim_hits = [h for h in hits if h["same_agent_set_and_type"]]
    return {
        "grain": "source_hash (corpus evidence text of each reviewed pair)",
        "n_demonstration_pairs": len(demos),
        "n_demonstration_claims": len({d["claim"] for d in demos}),
        "n_demonstration_sentences": len(by_sentence),
        "n_reviewed_pairs_scanned": scanned,
        "n_distinct_normalised_texts": len(reviewed_texts),
        "n_overlapping_pairs": len(hits),
        "n_overlapping_pairs_same_claim": len(same_claim_hits),
        "overlapping_pairs": [[h["paper_statement_hash"], h["source_hash"]] for h in hits],
        "hits": hits,
        "shared_token_count": len(shared),
        "vacuity_guard": "shared_token_count > 0",
        "disagrees_with_existing_agent_grain_check": len(same_claim_hits) > 0,
        "complementary_check": (
            "tests/test_paper_panel_fewshot_disjoint.py covers (agent set, statement "
            "type) grain; this is the evidence-pair grain the panel is scored at"
        ),
        "out_of_scope": (
            "pretraining contamination: the benchmark corpus and the paper repo are "
            "both public and this check cannot speak to that"
        ),
    }


def _normalise(text: str) -> str:
    return " ".join(text.casefold().split())


# --------------------------------------------------------------------------
# the same models, at statement grain
# --------------------------------------------------------------------------

def statement_grain_metrics(statement_gold: list, headtohead_path: Path) -> dict:
    """Each arm's statement-grain AUROC/AP, from its own shipped prediction file.

    Same models, same corpus, different unit of judgement: 1,689 assembled
    statements instead of 5,379 evidence pairs. Two grains on one axis is the
    figure, so both ends have to come from shipped artifacts. The reader arms'
    numbers are cross-checked against the statement head-to-head the rest of the
    page already reports; a disagreement gates rather than draws two different
    numbers for one model.
    """
    statement_ids = [r["canonical_corpus"]["statement_id"] for r in statement_gold]
    labels = np.array(
        [r["paper_replication_policy"]["released_paper_correct"] for r in statement_gold],
        dtype=np.int64,
    )
    shipped = json.loads(headtohead_path.read_text())["point_metrics"] if headtohead_path.is_file() else {}

    out: dict[str, dict] = {}
    for arm_id, relative in STATEMENT_PREDICTIONS.items():
        path = REPO / relative
        if not path.is_file():
            fail(f"statement_grain[{arm_id}]", f"{relative} is missing")
        table = {r["statement_id"]: float(r["probability_correct"]) for r in load_jsonl(path)}
        missing = [s for s in statement_ids if s not in table]
        if missing:
            fail(f"statement_grain[{arm_id}]", f"{len(missing)} statement(s) have no prediction")
        scores = np.array([table[s] for s in statement_ids], dtype=np.float64)
        auroc = float(roc_auc_score(labels, scores))
        ap = float(average_precision_score(labels, scores))
        crosscheck = None
        key = HEADTOHEAD_KEY_BY_ARM.get(arm_id)
        if key and key in shipped:
            delta_auroc = abs(auroc - float(shipped[key]["auroc"]))
            delta_ap = abs(ap - float(shipped[key]["pooled_average_precision"]))
            if max(delta_auroc, delta_ap) > STATEMENT_CROSSCHECK_TOLERANCE:
                fail(
                    f"statement_grain[{arm_id}]",
                    f"disagrees with the shipped statement head-to-head "
                    f"(ΔAUROC {delta_auroc}, ΔAP {delta_ap})",
                )
            crosscheck = {"key": key, "auroc_abs_diff": delta_auroc, "ap_abs_diff": delta_ap}
        out[arm_id] = {
            "n_statements": len(statement_ids),
            "auroc": auroc,
            "average_precision": ap,
            "distinct_scores": int(np.unique(scores).size),
            "predictions_path": relative,
            "predictions_sha256": sha256_file(path),
            "headtohead_crosscheck": crosscheck,
        }
    return {
        "positive_rate": float(labels.mean()),
        "label_rule": "paper_replication_policy.released_paper_correct",
        "note": (
            "A statement mark and an evidence mark are two measurements of ONE model on "
            "two different item populations (1,689 statements at a 73.2% positive rate "
            "vs 5,379 evidence pairs at 65.4%). The connector between them shows what "
            "changes when INDRA's noisy-OR turns per-evidence verdicts into a statement "
            "score; it is not a causal increment and the two AUROCs are not paired."
        ),
        "arms": out,
    }


# --------------------------------------------------------------------------
# markdown
# --------------------------------------------------------------------------

def render_markdown(artifact: dict) -> str:
    panel = artifact["panel"]
    reference = artifact["reference_arm_id"]
    by_id = {a["id"]: a for a in artifact["arms"]}
    lines = [
        "# Per-evidence comparison — the 2023 INDRA paper panel",
        "",
        f"**{panel['n_reviewed_pairs']} human-reviewed evidence pairs** "
        f"({panel['n_positive']} correct / {panel['n_negative']} incorrect, "
        f"{panel['negative_fraction']:.1%} negative) over "
        f"{panel['n_statements_with_reviewed_pair']} of {panel['n_statements']} statements. "
        f"Join key `(paper_statement_hash, source_hash)`.",
        "",
        f"Reference arm for every Δ: **{by_id[reference]['display']}** "
        f"(the strongest per-evidence baseline by AUROC).",
        "",
        "| Arm | kind | AUROC | AP(correct) | AP(incorrect) | err-F1 @0.5 | err-F1 best | distinct |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in artifact["arms"]:
        if "metrics" not in arm:
            lines.append(f"| {arm['display']} | {arm['kind']} | — | — | — | — | — | unavailable |")
            continue
        m = arm["metrics"]
        lines.append(
            f"| {arm['display']} | {arm['kind']} | {m['auroc']:.3f} | "
            f"{m['average_precision_correct']:.3f} | {m['average_precision_incorrect']:.3f} | "
            f"{m['error_detection']['f1']:.3f} | {m['error_detection_best_f1']:.3f} | "
            f"{m['distinct_scores']} |"
        )
    lines += ["", "## Paired Δ vs the reference (10,000 source-stratified resamples)", "",
              "| Arm | ΔAUROC [95% CI] | ΔAP(incorrect) [95% CI] |", "| --- | --- | --- |"]
    for arm_id, block in artifact["paired_delta_vs_reference"].items():
        auroc = block["pooled"]["auroc"]
        ap = block["pooled"]["average_precision_incorrect"]
        lines.append(
            f"| {by_id[arm_id]['display']} | "
            f"{auroc['mean']:+.3f} [{auroc['ci95_low']:+.3f}, {auroc['ci95_high']:+.3f}]"
            f"{'*' if auroc['excludes_zero'] else ''} | "
            f"{ap['mean']:+.3f} [{ap['ci95_low']:+.3f}, {ap['ci95_high']:+.3f}]"
            f"{'*' if ap['excludes_zero'] else ''} |"
        )
    lines += ["", "## Per source", "",
              "| Source | reviewed | correct frac | bundled prior @1 ev |",
              "| --- | --- | --- | --- |"]
    for row in artifact["coverage"]["sources"]:
        lines.append(
            f"| {row['source']} | {row['reviewed_pairs']} | "
            f"{row['observed_correct_fraction']:.3f} | "
            f"{row['bundled_prior_at_one_evidence']:.3f} |"
        )
    for block in artifact["shared_prior_defect"]["blocks"]:
        names = ", ".join(s["source"] for s in block["sources"])
        lines += [
            "",
            f"**Shared prior {block['shared_prior_at_one_evidence']:.3f} at one evidence** covers "
            f"{names}; observed correct fraction spans "
            f"{block['observed_correct_fraction_min']:.3f}–{block['observed_correct_fraction_max']:.3f} "
            f"(chi2={block['chi2']:.1f}, dof={block['dof']}, p={block['p_value']:.3g}).",
        ]
    reagg = artifact["reaggregation"]
    if reagg["verified"]:
        lines += ["", "## Grain bridge", "",
                  "Recovered per-evidence verdicts pushed back through "
                  "`statement_belief` reproduce the shipped statement probabilities:"]
        for arm_id, block in reagg["arms"].items():
            lines.append(
                f"- {by_id[arm_id]['display']}: {block['n_exact']}/{block['n_statements']} exact, "
                f"max |Δ| = {block['max_abs_diff']:g}"
            )
    lines += ["", "## Two grains, one model", "",
              "| Arm | per-evidence AUROC (n=5,379) | statement AUROC (n=1,689) | change |",
              "| --- | --- | --- | --- |"]
    for arm in artifact["arms"]:
        if "statement_grain" not in arm:
            continue
        evidence = arm["metrics"]["auroc"]
        statement = arm["statement_grain"]["auroc"]
        lines.append(
            f"| {arm['display']} | {evidence:.3f} | {statement:.3f} | {statement - evidence:+.3f} |"
        )
    lines += ["", f"_{artifact['statement_grain']['note']}_"]

    contamination = artifact["contamination"]
    lines += ["", "## Contamination at evidence-pair grain", "",
              f"{contamination['n_demonstration_sentences']} distinct demonstration sentences vs "
              f"{contamination['n_reviewed_pairs_scanned']} reviewed pairs: "
              f"**{contamination['n_overlapping_pairs']} overlapping pairs**, of which "
              f"**{contamination['n_overlapping_pairs_same_claim']}** also match the "
              f"demonstration claim at (agent set, statement type) grain."]
    sensitivity = contamination.get("sensitivity")
    if sensitivity:
        lines.append(
            f"Excluding them ({sensitivity['n_pairs_kept']} pairs kept) moves AUROC by at most "
            f"{sensitivity['max_auroc_abs_shift']:.4f}; the primary panel keeps every reviewed pair."
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as error:
        print(f"GATE FAILED — {error}", file=sys.stderr)
        raise SystemExit(2)
