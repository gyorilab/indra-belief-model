"""Derive slide-12 calibration and ship-gate numbers from production belief.

For each configuration-matched validation pair, this script:

* fits each reader profile on ITS OWN shipped fit corpus -- which is not the
  same corpus for every arm -- and selects hard/calibrated error-F1 thresholds
  there;
* joins the test run at production statement grain with any-incorrect-wins gold;
* reports the visible raw INDRA count -> calibrated ECE/AUROC comparison; and
* delegates the formal four-leg gate, including paired bootstrap CIs, to
  ``calibration_ship_gate``.

It is a read-only deck trace; the formal JSON/Markdown artifacts are generated
by ``scripts/calibration_ship_gate.py``.

Run: PYTHONPATH=src python scripts/deck_shipgate_fig.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import calibration_stage1 as c1  # noqa: E402
from calibration_ship_gate import (  # noqa: E402
    eval_reader,
    gate,
    statements_for_run,
    training_thresholds,
    validate_configuration_pair,
)
from indra_belief.metrics import auroc, ece  # noqa: E402
from indra_belief.noise_model import RECALIBRATED_PRIORS  # noqa: E402
from indra_belief.statement_belief import statement_belief  # noqa: E402

TRAIN_GOLD = "data/benchmark/eval_curation_v1.jsonl"
PLAN = [
    {
        "gold": "holdout_cc", "reader": "gemma-26B remote",
        "train_run": "data/results/eval_curation_v1_gemma.jsonl",
        "test_gold": "data/results/cc_holdout_cc/holdout_cc.jsonl",
        "test_run": "data/results/holdout_cc_gemma.jsonl",
    },
    {
        "gold": "holdout_cc", "reader": "MedPsy-4B",
        "train_run": "data/results/eval_curation_v1_medpsy.jsonl",
        "test_gold": "data/results/cc_holdout_cc/holdout_cc.jsonl",
        "test_run": "data/results/holdout_cc_medpsy.jsonl",
    },
    {
        "gold": "external", "reader": "gemma-26B Bedrock",
        # The external run uses the reasoning-first Bedrock configuration, whose
        # separately measured profile must not borrow the remote/Ollama fit.
        # REFIT 2026-08-17 (8bebc9a): the shipped gemma_bedrock_rf profile is
        # fitted on holdout_large_fit, not eval_curation_v1. This script kept
        # refitting on the old corpus and therefore kept printing the RETIRED
        # .129->.061 / .688->.814 that the deck cited it for -- a provenance
        # script certifying numbers production no longer uses.
        "train_gold": "data/benchmark/holdout_large_fit.jsonl",
        "train_run": "data/results/holdout_large_bedrock-gemma-4-26b_fit.jsonl",
        "test_gold": "data/benchmark/external_curator_gold_v1.jsonl",
        "test_run": "data/results/external_curator_v1_bedrock-gemma.jsonl",
    },
]


def raw_count_comparison(statements: list[dict], profile: dict) -> dict:
    """Paired raw INDRA count vs production calibrated belief metrics."""
    raw, calibrated, labels = [], [], []
    for stmt in statements:
        stored = stmt.get("stored_belief")
        soft = statement_belief(stmt["ev"], RECALIBRATED_PRIORS, soft=profile).belief
        if not isinstance(stored, (int, float)) or soft is None:
            continue
        raw.append(stored)
        calibrated.append(soft)
        labels.append(stmt["gold_correct"])
    return {
        "n": len(labels),
        "ece_raw": ece(list(zip(raw, labels))),
        "ece_calibrated": ece(list(zip(calibrated, labels))),
        "auroc_raw": auroc(raw, labels),
        "auroc_calibrated": auroc(calibrated, labels),
    }


def main() -> None:
    fitted = {}
    for item in PLAN:
        train_run = item["train_run"]
        train_gold = item.get("train_gold", TRAIN_GOLD)
        validate_configuration_pair(train_run, item["test_run"])
        if train_run not in fitted:
            train_statements, _ = statements_for_run(train_run, train_gold)
            profile = c1.fit_reader_profile(train_statements)
            fitted[train_run] = (profile, training_thresholds(train_statements, profile))

    print(
        f"{'gold':12} {'reader':20} {'n':>4}  {'raw ECE->cal':>16}  "
        f"{'raw AUC->cal':>16}  {'hard ECE->cal':>16}  {'hard AUC->cal':>16}  "
        f"{'errF1 delta [CI]':>24}  legs"
    )
    for item in PLAN:
        profile, thresholds = fitted[item["train_run"]]
        statements, join = statements_for_run(item["test_run"], item["test_gold"])
        comparison = raw_count_comparison(statements, profile)
        evaluation = eval_reader(statements, profile, thresholds, join)
        # This trace is run only as part of the verified deck build, after the
        # E4 compatibility test. The formal artifact requires the equivalent
        # explicit --e4-identity-pass flag.
        verdict = gate(evaluation, e4_identity_pass=True)
        legs = "".join(
            "✓" if verdict[key]["pass"] else "✗"
            for key in ("ece", "auroc", "errf1", "e4_identity")
        )
        ci = verdict["errf1"]["ci_delta"]
        print(
            f"{item['gold']:12} {item['reader']:20} {comparison['n']:>4}  "
            f"{comparison['ece_raw']:.3f}->{comparison['ece_calibrated']:.3f}      "
            f"{comparison['auroc_raw']:.3f}->{comparison['auroc_calibrated']:.3f}      "
            f"{verdict['ece']['hard']:.3f}->{verdict['ece']['soft']:.3f}      "
            f"{verdict['auroc']['hard']:.3f}->{verdict['auroc']['soft']:.3f}      "
            f"{verdict['errf1']['delta']:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  "
            f"{legs} ({sum(verdict[k]['pass'] for k in ('ece', 'auroc', 'errf1', 'e4_identity'))}/4)"
        )


if __name__ == "__main__":
    main()
