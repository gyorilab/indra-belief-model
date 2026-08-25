"""Cross-check tracked calibration decisions against production profiles."""
from __future__ import annotations

import json
from pathlib import Path

from indra_belief.calibration_constants import fitted_calibration_for

ROOT = Path(__file__).resolve().parents[1]
# One artifact per (reader, validation gold). The gate script defaults
# --json to calibration_ship_gate.json, so a run against a non-default
# --test-gold will overwrite the holdout_cc gate unless it is given its own
# path — which is how the external and local files below came to exist.
ARTIFACTS = (
    ROOT / "data/results/calibration_ship_gate.json",
    ROOT / "data/results/calibration_ship_gate_external.json",
    ROOT / "data/results/calibration_ship_gate_local.json",
)


def _configuration_parts(configuration: str) -> tuple[str, str]:
    model, prompt = configuration.split("@prompt-sha256:", 1)
    assert len(prompt) == 64
    return model, prompt


def test_decision_artifacts_name_the_exact_production_profiles():
    seen = set()
    for artifact in ARTIFACTS:
        for row in json.loads(artifact.read_text()):
            profile = row["reader_profile"]
            provenance = row["provenance"]
            model, prompt = _configuration_parts(profile["reader_configuration"])
            production = fitted_calibration_for(model, prompt_sha256=prompt)
            assert production is not None

            assert profile["profile_id"] == production["profile_id"]
            assert profile["reader_configuration"] == production["reader_configuration"]
            assert profile["confusion"] == production["confusion"]
            assert profile["fit_gold"] == production["fit_gold"]
            assert profile["fit_gold_sha256"] == production["fit_gold_sha256"]

            validation = production["validation"]
            assert provenance["train_gold_sha256"] == production["fit_gold_sha256"]
            assert provenance["test_gold"] == validation["gold"]
            assert provenance["test_gold_sha256"] == validation["gold_sha256"]
            assert provenance["test_run"] == validation["run"]
            assert validation["result"] == ("pass" if row["gate"]["overall"] else "fail")
            assert production["deployment_status"] == (
                "enabled" if row["gate"]["overall"] else "disabled"
            )
            assert len(provenance["train_run_sha256"]) == 64
            assert len(provenance["test_run_sha256"]) == 64
            seen.add(profile["profile_id"])

    # One per gate row: medpsy-4b + gemma-26B (holdout_cc), bedrock-gemma-26B
    # (external_curator_gold_v1), local-gemma-26B (external_curator_gold_v1).
    assert len(seen) == 4
