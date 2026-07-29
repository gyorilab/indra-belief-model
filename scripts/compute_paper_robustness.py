"""How the margin behaves under two standard robustness checks.

The head-to-head (``scripts/compare_paper_literal_vs_llms.py``) reports, per arm,
a POINT estimate and a POINTWISE 95% interval against the paper's own re-run RF.
On the paper's exact 1689-statement panel with the paper's exact released labels
(``paper_replication_policy.released_paper_correct``, unmodified) the Gemma 4 26B
gate beats that RF on tie-robust pooled average precision.  That result is the
primary one and nothing here replaces it.

Two questions that a referee will ask about it are not answerable from a
pointwise interval, and this script answers both.

(A) MULTIPLICITY.  ``data/comparison/run_plan.json`` — frozen 2026-07-22, before
    the comparison bundles were generated — stages FOUR reader arms (e2b,
    gemma_26b, gemma_31b, glm_5) and designates none of them as the confirmatory
    arm.  We therefore cannot claim to have pre-registered one arm, and a
    simultaneous band over the family of four is a fair ask.  The arms are
    strongly rank-correlated, so the correct correction is much milder than
    Bonferroni: a studentized max-t band computed from the SAME paired bootstrap
    draws costs ~2.30 standard errors where Bonferroni would cost ~2.50 and a
    pointwise normal interval costs 1.96.

(B) LABEL COMPLETENESS.  111 of the panel's 452 negatives carry
    ``paper_replication_policy.label_is_adjudication_safe == false``: every one is
    a negative whose evidence review is incomplete.  They ARE negative in the
    paper's released labels, so dropping them is OUR revision of THEIR labels.
    It is a sensitivity check and never the primary result — and it is not a free
    one: it removes 24.6% of all negatives and moves the panel from 26.8% to
    21.6% negative, so it changes what the panel is a sample of, not only its
    label quality.

REPRODUCTION IS THE CONTRACT.  The bootstrap here is not a re-implementation: the
panel, the fold-stratified resampling design, the seed, the resample count and
the AP call are imported from / mirrored on the head-to-head, and the script
ASSERTS that every pointwise number it recomputes reproduces the shipped
``paired_delta_vs_paper_literal`` value to ``TOL_SHIPPED``.  If it does not, the
simultaneous band would be describing a different bootstrap than the interval it
is being compared against, so the script exits non-zero instead of writing.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/compute_paper_robustness.py \
      --literal data/results/indra_paper_literal_models_20260724/paper_literal_table6_and_oof.json \
      --comparison data/results/indra_paper_literal_models_20260724/paper_literal_vs_llms.json \
      --out-json data/results/indra_paper_literal_models_20260724/paper_margin_robustness.json \
      --manifest data/results/indra_paper_literal_models_20260724/manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The join contract, the resampling design and the metric all come from the
# head-to-head verbatim.  Importing them is what makes "reproduces the shipped
# numbers" a property of one implementation rather than of two that agree today.
from compare_paper_literal_vs_llms import (  # noqa: E402
    GOLD,
    HEADLINE,
    MODELS_DIR,
    N_BOOT,
    SEED,
    load_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]

RUN_PLAN = "data/comparison/run_plan.json"

# The paired-delta baseline: the paper's own released RF code, re-run by us on
# their panel.  `label` is the FROZEN point_metrics join key; the on-screen name
# lives in viewer/src/lib/data/paper-robustness.ts and is never read from here.
REFERENCE_ID = "paper-rf-promoter"
REFERENCE_LABEL = "Paper literal RF+promoter"

# The multiplicity family: every reader arm the frozen run plan stages, and
# nothing else.  `stage` is the run_plan stage id, `bundle` the prediction
# directory under data/comparison/models, `label` the frozen point_metrics key.
READER_ARMS = [
    {"id": "gemma-4-e2b", "label": "Gemma 4 E2B", "stage": "e2b", "bundle": "gemma_4_e2b"},
    {"id": "gemma-4-26b", "label": "Gemma 4 26B", "stage": "gemma_26b", "bundle": "gemma_4_26b"},
    {"id": "gemma-4-31b", "label": "Gemma 4 31B", "stage": "gemma_31b", "bundle": "gemma_4_31b"},
    {"id": "glm-5", "label": "GLM-5", "stage": "glm_5", "bundle": "glm_5"},
]

# The one arm we are willing to call SMALLEST on the record: Gemma 4 E2B is the
# family's edge variant.  No further size ordering is claimed — GLM-5's parameter
# count is not published, so "26B < 31B < GLM-5" would be an invention.  The
# falsifiable half of the dose-response observation is asserted below instead:
# exactly one arm has a negative delta, and it is this one.
SMALLEST_ARM_ID = "gemma-4-e2b"

# Family-wise level for the simultaneous band. The two reference critical values
# it is quoted against are DERIVED, not typed: a transcribed normal quantile is
# exactly the kind of constant that is wrong in the fourth decimal and never
# noticed. `statistics.NormalDist` is stdlib, so this adds no dependency.
FAMILY_ALPHA = 0.05
# Pointwise two-sided normal quantile, z_{1 - alpha/2}.
POINTWISE_NORMAL_Z = statistics.NormalDist().inv_cdf(1 - FAMILY_ALPHA / 2)

# Every pointwise number recomputed here must reproduce the shipped one this
# tightly.  The two computations are the same operations in the same order on
# the same draws, so the achievable residual is 0.0; 1e-12 leaves room for a
# libm difference without leaving room for a design difference.
TOL_SHIPPED = 1e-12


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mid_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged.

    The head-to-head's own `_spearman` uses argsort-of-argsort, which breaks ties
    arbitrarily. That is harmless there (it compares two near-continuous RF score
    vectors) and wrong here: the reader arms emit only ~420-498 distinct scores
    over 1689 statements, so an arbitrary tie order would invent rank agreement
    or disagreement that the scores do not contain.
    """
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start
        while stop + 1 < len(values) and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop + 1]] = (start + stop) / 2
        start = stop + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Tie-aware rank correlation, no scipy dependency."""
    return float(np.corrcoef(_mid_ranks(a), _mid_ranks(b))[0, 1])


def fold_stratified_indices(folds: np.ndarray, n_boot: int, seed: int) -> list[np.ndarray]:
    """The head-to-head's resampling design, unchanged.

    ONE index vector per resample, SHARED across every arm — that pairing is what
    makes the deltas comparable, and it is also what makes the max-t band valid:
    the four arms' bootstrap deltas are drawn jointly, so their correlation is
    carried rather than assumed away.
    """
    rng = np.random.default_rng(seed)
    fold_ids = sorted(set(folds.tolist()))
    idx_by_fold = {f: np.where(folds == f)[0] for f in fold_ids}
    return [
        np.concatenate(
            [rng.choice(idx_by_fold[f], size=len(idx_by_fold[f]), replace=True) for f in fold_ids]
        )
        for _ in range(n_boot)
    ]


def bootstrap_deltas(
    y: np.ndarray, arm: np.ndarray, base: np.ndarray, boot_idx: list[np.ndarray]
) -> np.ndarray:
    """ΔAP (arm − reference) on each resample, exactly as the head-to-head does it."""
    out = []
    for take in boot_idx:
        yb = y[take]
        if len(set(yb.tolist())) == 2:
            out.append(
                average_precision_score(yb, arm[take]) - average_precision_score(yb, base[take])
            )
    return np.array(out)


class Panel:
    """One evaluation panel: a labelled subset of the 1689 statements.

    No model is refit and no score is recomputed between panels — the sensitivity
    panel scores the SAME prediction vectors on fewer statements.  That is the
    whole point: the only thing that changes is which labels are admitted.
    """

    def __init__(self, panel_id: str, mask: np.ndarray, y: np.ndarray, folds: np.ndarray,
                 probs: dict[str, np.ndarray]):
        self.id = panel_id
        self.mask = mask
        self.y = y[mask]
        self.folds = folds[mask]
        self.probs = {name: vector[mask] for name, vector in probs.items()}
        self.n = int(mask.sum())
        self.n_positive = int((self.y == 1).sum())
        self.n_negative = int((self.y == 0).sum())
        self.boot_idx = fold_stratified_indices(self.folds, N_BOOT, SEED)
        self.base = self.probs[REFERENCE_LABEL]
        self.reference_ap = float(average_precision_score(self.y, self.base))

    def census(self) -> dict:
        return {
            "n_statements": self.n,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "negative_fraction": self.n_negative / self.n,
            "n_folds": len(set(self.folds.tolist())),
            "reference_average_precision": self.reference_ap,
        }

    def arm_result(self, label: str) -> dict:
        p = self.probs[label]
        deltas = bootstrap_deltas(self.y, p, self.base, self.boot_idx)
        ap = float(average_precision_score(self.y, p))
        return {
            "average_precision": ap,
            "delta": ap - self.reference_ap,
            "delta_bootstrap_mean": float(deltas.mean()),
            "ci95_low": float(np.percentile(deltas, 2.5)),
            "ci95_high": float(np.percentile(deltas, 97.5)),
            "p_greater_than_zero": float((deltas > 0).mean()),
            "bootstrap_se": float(deltas.std(ddof=1)),
            "n_valid_resamples": int(len(deltas)),
            "_draws": deltas,
        }


def simultaneous_band(results: dict[str, dict]) -> tuple[float, dict[str, tuple[float, float]]]:
    """Studentized max-t critical value over the family, and each arm's band.

    t_bj = (delta_bj − delta_j) / se_j on the SHARED draws; the critical value is
    the 95th percentile of max_j |t_bj|.  Because the four arms move together
    from resample to resample, that maximum is far smaller than four independent
    tests would give — which is why this correction is milder than Bonferroni and
    why quoting Bonferroni here would be the conservative-looking wrong answer.
    """
    labels = list(results)
    # The band is only "simultaneous" if every arm was measured on the SAME
    # resamples. A resample where one arm's slice went single-class would drop
    # that arm's draw and silently misalign the columns, so it is caught here
    # rather than discovered as a shape error.
    widths = {label: len(results[label]["_draws"]) for label in labels}
    assert len(set(widths.values())) == 1, (
        f"arms were measured on different numbers of resamples ({widths}); the max-t "
        f"band requires one shared draw per arm per resample")
    draws = np.column_stack([results[label]["_draws"] for label in labels])
    centers = np.array([results[label]["delta"] for label in labels])
    ses = np.array([results[label]["bootstrap_se"] for label in labels])
    t = np.abs((draws - centers) / ses)
    critical = float(np.percentile(t.max(axis=1), 100 * (1 - FAMILY_ALPHA)))
    bands = {
        label: (float(centers[i] - critical * ses[i]), float(centers[i] + critical * ses[i]))
        for i, label in enumerate(labels)
    }
    return critical, bands


def excludes_zero(low: float, high: float) -> bool:
    return bool(low > 0 or high < 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--literal", required=True,
                    help="paper_literal_table6_and_oof.json from run_indra_paper_literal_models.py")
    ap.add_argument("--comparison", required=True,
                    help="paper_literal_vs_llms.json — the shipped pointwise result to reproduce")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--manifest", default=None,
                    help="run manifest.json to record the output path + sha256 in")
    args = ap.parse_args()

    # ---- panel: the head-to-head's join, unchanged ---------------------------
    lit = json.load(open(args.literal))
    oof = {r["stmt_hash"]: r for r in lit["oof_predictions"][HEADLINE]}

    gold = {}
    for r in load_jsonl(GOLD):
        policy = r["paper_replication_policy"]
        gold[int(r["paper_statement_hash"])] = {
            "sid": r["canonical_corpus"]["statement_id"],
            "label": policy["released_paper_correct"],
            "adjudication_safe": bool(policy["label_is_adjudication_safe"]),
            "negative_has_complete_evidence": policy["negative_has_complete_evidence"],
        }

    hashes = sorted(oof)
    sids = [gold[h]["sid"] for h in hashes]
    y = np.array([oof[h]["y_true"] for h in hashes])
    folds = np.array([oof[h]["fold_ix"] for h in hashes])
    assert all(oof[h]["y_true"] == gold[h]["label"] for h in hashes), "label mismatch"

    probs = {REFERENCE_LABEL: np.array([oof[h]["prob_correct"] for h in hashes])}
    for arm in READER_ARMS:
        bundle = {r["statement_id"]: r["probability_correct"]
                  for r in load_jsonl(f"{MODELS_DIR}/{arm['bundle']}/all_source_predictions.jsonl")}
        probs[arm["label"]] = np.array([bundle[s] for s in sids])

    # ---- the label-completeness split ---------------------------------------
    safe = np.array([gold[h]["adjudication_safe"] for h in hashes])
    dropped = ~safe
    n_dropped = int(dropped.sum())
    # The check is only meaningful if the dropped rows are what the field says
    # they are.  Assert it rather than describe it.
    assert (y[dropped] == 0).all(), "an adjudication-unsafe statement is not a negative"
    assert all(gold[h]["negative_has_complete_evidence"] is False
               for h, drop in zip(hashes, dropped) if drop), \
        "an adjudication-unsafe statement claims complete evidence review"

    primary = Panel("paper_labels_1689", np.ones(len(hashes), dtype=bool), y, folds, probs)
    sensitivity = Panel("adjudication_safe_1578", safe, y, folds, probs)
    assert primary.n_positive == sensitivity.n_positive, \
        "the label-completeness check must drop negatives only"

    # ---- pointwise reproduction of the shipped head-to-head ------------------
    shipped = json.load(open(args.comparison))
    shipped_deltas = shipped["paired_delta_vs_paper_literal"]
    shipped_points = shipped["point_metrics"]

    reference_residual = abs(
        primary.reference_ap - shipped_points[REFERENCE_LABEL]["pooled_average_precision"]
    )
    assert reference_residual <= TOL_SHIPPED, (
        f"reference average precision {primary.reference_ap} does not reproduce the shipped "
        f"{shipped_points[REFERENCE_LABEL]['pooled_average_precision']}")

    primary_results = {}
    sensitivity_results = {}
    reconciliation = {}
    for arm in READER_ARMS:
        label = arm["label"]
        primary_results[label] = primary.arm_result(label)
        sensitivity_results[label] = sensitivity.arm_result(label)

        want = shipped_deltas[label]["pooled_average_precision"]
        got = primary_results[label]
        residuals = {
            "delta": abs(got["delta_bootstrap_mean"] - want["delta"]),
            "ci95_low": abs(got["ci95_low"] - want["ci95_low"]),
            "ci95_high": abs(got["ci95_high"] - want["ci95_high"]),
            "p_arm_greater": abs(got["p_greater_than_zero"] - want["p_arm_greater"]),
            "average_precision": abs(
                got["average_precision"] - shipped_points[label]["pooled_average_precision"]
            ),
        }
        worst = max(residuals.values())
        assert got["n_valid_resamples"] == want["n_valid_resamples"], label
        assert worst <= TOL_SHIPPED, (
            f"{label}: recomputed pointwise result does not reproduce the shipped one "
            f"(worst residual {worst:.3e} > {TOL_SHIPPED:.0e}); the simultaneous band would "
            f"describe a different bootstrap than the interval it is compared against")
        reconciliation[label] = {
            "worst_residual_vs_shipped": worst,
            "per_field_residual": residuals,
        }

    critical, primary_bands = simultaneous_band(primary_results)
    sensitivity_critical, sensitivity_bands = simultaneous_band(sensitivity_results)
    # Bonferroni over |family| two-sided tests, for scale. Derived from the family
    # size actually used, so it cannot drift if the family ever changes.
    bonferroni_z = statistics.NormalDist().inv_cdf(1 - FAMILY_ALPHA / (2 * len(READER_ARMS)))
    assert POINTWISE_NORMAL_Z < critical <= bonferroni_z, (
        f"max-t critical value {critical:.4f} is outside "
        f"({POINTWISE_NORMAL_Z:.4f}, {bonferroni_z:.4f}] — a simultaneous band cannot be "
        f"narrower than pointwise nor wider than Bonferroni")

    # ---- correlation structure that makes max-t mild ------------------------
    pairs = []
    for i, a in enumerate(READER_ARMS):
        for b in READER_ARMS[i + 1:]:
            pairs.append({
                "a": a["id"],
                "b": b["id"],
                "score_spearman": spearman(probs[a["label"]], probs[b["label"]]),
                "bootstrap_delta_pearson": float(
                    np.corrcoef(primary_results[a["label"]]["_draws"],
                                primary_results[b["label"]]["_draws"])[0, 1]
                ),
            })
    score_rhos = [p["score_spearman"] for p in pairs]

    # ---- frozen run-plan provenance for the multiplicity claim --------------
    plan_path = ROOT / RUN_PLAN
    plan = json.loads(plan_path.read_text())
    staged = {stage["id"]: stage for stage in plan["stages"]}
    assert set(staged) == {arm["stage"] for arm in READER_ARMS}, (
        "the frozen run plan does not stage exactly the four arms in this family")
    # "No designated primary arm" is a checkable property of the plan, not a
    # reading of it: no stage record carries a field that singles one arm out.
    designating = sorted(
        key for stage in plan["stages"] for key in stage
        if any(token in key.lower() for token in ("primary", "confirmatory", "preregister"))
    )
    assert not designating, f"the run plan designates an arm after all: {designating}"

    # ---- assemble ------------------------------------------------------------
    def arm_payload(arm: dict) -> dict:
        label = arm["label"]

        def side(result: dict, band: tuple[float, float], panel: Panel) -> dict:
            return {
                "panel": panel.id,
                "average_precision": result["average_precision"],
                "delta": result["delta"],
                "delta_bootstrap_mean": result["delta_bootstrap_mean"],
                "ci95_low": result["ci95_low"],
                "ci95_high": result["ci95_high"],
                "p_greater_than_zero": result["p_greater_than_zero"],
                "bootstrap_se": result["bootstrap_se"],
                "n_valid_resamples": result["n_valid_resamples"],
                "simultaneous_low": band[0],
                "simultaneous_high": band[1],
                "excludes_zero_pointwise": excludes_zero(result["ci95_low"], result["ci95_high"]),
                "excludes_zero_simultaneous": excludes_zero(band[0], band[1]),
            }

        return {
            "id": arm["id"],
            "label": label,
            "run_plan_stage": arm["stage"],
            "provider_model_id": staged[arm["stage"]]["provider_model_id"],
            "prediction_bundle": f"{MODELS_DIR}/{arm['bundle']}/all_source_predictions.jsonl",
            "primary": side(primary_results[label], primary_bands[label], primary),
            "sensitivity": side(sensitivity_results[label], sensitivity_bands[label], sensitivity),
            "shipped_reconciliation": reconciliation[label],
        }

    arms = [arm_payload(arm) for arm in READER_ARMS]
    # Deterministic presentation order: strongest primary margin first, so the
    # arm that loses sits at the bottom of the figure rather than being buried.
    arms.sort(key=lambda a: (-a["primary"]["delta"], a["id"]))

    negatives = [a for a in arms if a["primary"]["delta"] < 0]
    assert len(negatives) == 1 and negatives[0]["id"] == SMALLEST_ARM_ID, (
        "the dose-response observation no longer holds: the set of arms with a negative "
        "primary delta is not exactly {" + SMALLEST_ARM_ID + "}")

    payload = {
        "artifact_kind": "paper_margin_robustness",
        "metric": "pooled average precision (sklearn average_precision_score)",
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "bootstrap_design": (
            "paired fold-stratified bootstrap over the paper's own out-of-fold fold "
            "assignment; one index vector per resample, SHARED across all arms; deltas are "
            "arm minus the paper's re-run RF on the same resample. Imported from "
            "scripts/compare_paper_literal_vs_llms.py."
        ),
        "reference": {
            "id": REFERENCE_ID,
            "label": REFERENCE_LABEL,
            "description": (
                "our re-run of the paper's own released RF code (sorgerlab/indra_assembly_paper), "
                "scored out-of-fold on the paper's own folds"
            ),
        },
        "panels": {
            "primary": {
                **primary.census(),
                "id": primary.id,
                "role": "primary",
                "label_field": "paper_replication_policy.released_paper_correct",
                "label_provenance": "the 2023 paper's released labels, unmodified",
                "is_our_label_revision": False,
            },
            "sensitivity": {
                **sensitivity.census(),
                "id": sensitivity.id,
                "role": "sensitivity",
                "label_field": "paper_replication_policy.released_paper_correct",
                "label_provenance": (
                    "the paper's released labels with 111 statements REMOVED by us — every one "
                    "of them a negative in the paper's labels whose evidence review is "
                    "incomplete (label_is_adjudication_safe == false)"
                ),
                "is_our_label_revision": True,
            },
        },
        "label_completeness": {
            "field": "paper_replication_policy.label_is_adjudication_safe",
            "n_dropped": n_dropped,
            "all_dropped_are_negative": True,
            "all_dropped_have_incomplete_evidence_review": True,
            "dropped_share_of_all_negatives": n_dropped / primary.n_negative,
            "dropped_share_of_panel": n_dropped / primary.n,
            "negative_fraction_before": primary.n_negative / primary.n,
            "negative_fraction_after": sensitivity.n_negative / sensitivity.n,
            "no_model_is_refit": True,
            "note": (
                "These 111 statements are NEGATIVE in the paper's released labels. Dropping them "
                "is our revision of their labels, so this is a sensitivity check and never the "
                "primary result. It also removes a quarter of all negatives and shifts the class "
                "balance, so it changes what the panel is a sample of, not only its label quality."
            ),
        },
        "multiplicity": {
            "family": [arm["id"] for arm in arms],
            "family_size": len(arms),
            "method": "studentized max-t over the shared paired-bootstrap draws",
            "family_alpha": FAMILY_ALPHA,
            "critical_value": critical,
            "sensitivity_critical_value": sensitivity_critical,
            "pointwise_normal_critical_value": POINTWISE_NORMAL_Z,
            "bonferroni_critical_value": bonferroni_z,
            "score_spearman_min": min(score_rhos),
            "score_spearman_max": max(score_rhos),
            "pairwise_correlation": pairs,
            "no_designated_primary_arm": True,
            "run_plan": {
                "path": RUN_PLAN,
                "sha256": sha256_of(plan_path),
                "frozen_at": plan["amendment"]["frozen_at"],
                "stages": [stage["id"] for stage in plan["stages"]],
            },
            "note": (
                "The frozen run plan stages all four reader arms and designates none of them as "
                "the confirmatory one, so we cannot claim a pre-registered single arm and a "
                "simultaneous band over the four is a fair ask. The arms are strongly "
                "rank-correlated, so max-t costs far less than Bonferroni would."
            ),
        },
        "dose_response": {
            "smallest_arm_id": SMALLEST_ARM_ID,
            "n_arms_with_negative_delta": len(negatives),
            "n_arms_with_positive_delta": len(arms) - len(negatives),
            "only_negative_arm_is_the_smallest": True,
            "basis": (
                "Gemma 4 E2B is the family's edge variant and the smallest arm; the other three "
                "are larger. No finer size ordering is claimed — GLM-5's parameter count is not "
                "published. The checkable part is the sign pattern: exactly one arm is negative "
                "and it is the smallest."
            ),
        },
        "power": {
            "pointwise_ci_half_width": {
                arm["id"]: (arm["primary"]["ci95_high"] - arm["primary"]["ci95_low"]) / 2
                for arm in arms
            },
            "note": (
                "The pointwise half-width is the same order as the effect it is measuring, so "
                "this panel is underpowered for an effect this size rather than silent about it."
            ),
        },
        "arms": arms,
        "shipped_reconciliation": {
            "comparison_artifact": args.comparison,
            "tolerance": TOL_SHIPPED,
            "reference_average_precision_residual": reference_residual,
            "worst_residual_vs_shipped": max(
                r["worst_residual_vs_shipped"] for r in reconciliation.values()
            ),
            "fields_reproduced": [
                "paired_delta_vs_paper_literal[arm].pooled_average_precision.delta",
                "paired_delta_vs_paper_literal[arm].pooled_average_precision.ci95_low",
                "paired_delta_vs_paper_literal[arm].pooled_average_precision.ci95_high",
                "paired_delta_vs_paper_literal[arm].pooled_average_precision.p_arm_greater",
                "point_metrics[arm].pooled_average_precision",
            ],
        },
        "inputs": {
            "literal": args.literal,
            "comparison": args.comparison,
            "gold": GOLD,
            "model_bundles": f"{MODELS_DIR}/{{arm}}/all_source_predictions.jsonl",
            "join": "paper_statement_hash -> canonical_corpus.statement_id",
            "generated_by": "scripts/compute_paper_robustness.py",
        },
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")

    sha = sha256_of(out)
    if args.manifest:
        mpath = Path(args.manifest)
        man = json.loads(mpath.read_text())
        man.setdefault("outputs", {})["margin_robustness"] = out.name
        man.setdefault("output_sha256", {})[out.name] = sha
        mpath.write_text(json.dumps(man, indent=2) + "\n")
        print(f"recorded sha256 in {mpath}")

    print(f"\nwrote {out} ({out.stat().st_size} bytes)\nsha256 {sha}\n")
    print(f"panel 1689 (paper labels, PRIMARY): reference AP {primary.reference_ap:.4f} · "
          f"{primary.n_positive}/{primary.n_negative} · "
          f"{primary.n_negative / primary.n:.1%} negative")
    print(f"panel {sensitivity.n} (our label revision, sensitivity): reference AP "
          f"{sensitivity.reference_ap:.4f} · {sensitivity.n_positive}/{sensitivity.n_negative} · "
          f"{sensitivity.n_negative / sensitivity.n:.1%} negative")
    print(f"max-t critical value {critical:.4f} (pointwise normal {POINTWISE_NORMAL_Z:.4f}, "
          f"Bonferroni {bonferroni_z:.4f}); score Spearman "
          f"{min(score_rhos):.2f}-{max(score_rhos):.2f}\n")
    print(f"{'arm':<14}{'dAP':>9}{'pointwise 95%':>22}{'simultaneous max-t':>24}{'1578':>9}")
    for arm in arms:
        p = arm["primary"]
        s = arm["sensitivity"]
        pointwise = "[{:+.4f}, {:+.4f}]".format(p["ci95_low"], p["ci95_high"])
        simultaneous = "[{:+.4f}, {:+.4f}]".format(p["simultaneous_low"], p["simultaneous_high"])
        print(f"{arm['label']:<14}{p['delta']:>+9.4f}{pointwise:>22}"
              f"{simultaneous:>24}{s['delta']:>+9.4f}")


if __name__ == "__main__":
    main()
