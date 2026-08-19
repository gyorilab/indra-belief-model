"""The belief table publishes `{matches_hash: belief}` without lying about it.

The table IS the deployment: everything downstream of INDRA reads a frozen copy
of that column. Three failure modes would each be invisible in the artifact
itself, so they are pinned here.

  1. A missing belief published as a number. `Statement.from_json` defaults a
     MISSING belief to 1.0, so an unscored statement must be ABSENT, not filled.
     Emitting a placeholder would publish "certainly true" for something never
     read — and it would look completely ordinary in the file.

  2. A hash collision resolved silently. Two statements on one matches_hash with
     different beliefs is an upstream join defect; last-writer-wins buries it.

  3. A calibrated table labelled uncalibrated (or the reverse). The number is
     meaningless without the profile that produced it, and the label is the only
     place a consumer can read that. This one is not hypothetical: the first
     implementation read a top-level `soft_weights` key that is always None, and
     labelled a fitted table "HARD GATE".
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from export_belief_table import build_table  # noqa: E402


def test_unscored_statements_are_omitted_not_defaulted():
    rows = [
        {"indra_matches_hash": "1", "belief": 0.9},
        {"indra_matches_hash": "2", "belief": None},      # nothing was read
        {"indra_matches_hash": "3", "belief": 0.1},
    ]
    table, diag = build_table(rows)
    assert "2" not in table, "an unscored statement must not appear in the table"
    assert 1.0 not in table.values(), "no statement may be defaulted to certainty"
    assert diag["n_unscored_omitted"] == 1
    assert diag["n_published"] == 2


def test_a_zero_belief_is_published_not_treated_as_missing():
    """0.0 is a real, strong claim and must survive the None check."""
    table, diag = build_table([{"indra_matches_hash": "1", "belief": 0.0}])
    assert table == {"1": 0.0}
    assert diag["n_unscored_omitted"] == 0


def test_statements_without_a_matches_hash_are_counted_not_dropped_silently():
    table, diag = build_table([
        {"indra_matches_hash": None, "belief": 0.5},
        {"indra_matches_hash": "1", "belief": 0.5},
    ])
    assert diag["n_without_matches_hash"] == 1
    assert diag["n_published"] == 1


def test_conflicting_beliefs_on_one_hash_are_reported():
    """Silent last-writer-wins would hide an upstream join defect."""
    table, diag = build_table([
        {"indra_matches_hash": "1", "belief": 0.9},
        {"indra_matches_hash": "1", "belief": 0.2},
    ])
    assert diag["n_hash_collisions_with_differing_belief"] == 1
    assert "1" in diag["collision_examples"]


def test_identical_beliefs_on_one_hash_are_not_a_collision():
    _, diag = build_table([
        {"indra_matches_hash": "1", "belief": 0.9},
        {"indra_matches_hash": "1", "belief": 0.9},
    ])
    assert diag["n_hash_collisions_with_differing_belief"] == 0


@pytest.mark.skipif(
    not (ROOT / "data/results/eval_curation_v1_gemma_rf_bedrock.jsonl").exists()
    or not (ROOT / "data/corpora/eval_curation_v1_statements.json").exists(),
    reason="needs the gitignored run/corpus tree",
)
def test_end_to_end_manifest_names_the_profile_that_produced_the_numbers(tmp_path):
    """The regression that motivated the guard: a fitted table labelled HARD GATE."""
    out = tmp_path / "table.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/export_belief_table.py"),
         "--run", "data/results/eval_curation_v1_gemma_rf_bedrock.jsonl",
         "--corpus", "data/corpora/eval_curation_v1_statements.json",
         "--model", "bedrock-gemma-4-26b", "--run-id", "test",
         "--require-calibrated", "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    manifest = json.loads(out.with_suffix(".manifest.json").read_text())
    assert manifest["calibrated"] is True
    assert manifest["profile_id"], "a calibrated table must name its profile"
    assert manifest["prompt_sha256"] and len(manifest["prompt_sha256"]) == 64
    table = json.loads(out.read_text())
    assert table and all(0.0 <= v <= 1.0 for v in table.values())
