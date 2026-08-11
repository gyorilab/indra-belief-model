"""Known-answer and contract tests for the held-out probe-battery evaluator."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from indra_belief import metrics


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval_probe_battery.py"
REAL_GOLD = ROOT / "data" / "results" / "cc_holdout_cc" / "holdout_cc.jsonl"
sys.path.insert(0, str(ROOT / "scripts"))

import eval_probe_battery as epb  # noqa: E402
from compute_deployed_baseline_replication import (  # noqa: E402
    paired_bootstrap_delta as sibling_paired_bootstrap_delta,
)


SHARED_I = np.arange(120)
SHARED_LABELS = np.array(
    [(int(i) * 17) % 23 < 11 for i in SHARED_I], dtype=bool
)
SHARED_CANDIDATE = ((SHARED_I * 7) % 13) / 12
SHARED_INCUMBENT = ((SHARED_I * 11) % 6) / 5
SHARED_CLUSTER_IDS = [f"{i:04d}" for i in range(len(SHARED_I))]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _pairs(
    labels,
    candidate,
    incumbent,
    *,
    candidate_seconds=None,
    incumbent_seconds=None,
    cluster_ids=None,
    cluster_field="fixture_cluster",
) -> epb.Pairs:
    labels = list(labels)
    n = len(labels)
    if cluster_ids is None:
        cluster_ids = [f"{i:04d}" for i in range(n)]
    return epb.Pairs(
        record_ids=[f"{i}:fixture-{i}" for i in range(n)],
        cluster_ids=list(cluster_ids),
        cluster_field=cluster_field,
        labels=[bool(value) for value in labels],
        candidate=[float(value) for value in candidate],
        incumbent=[float(value) for value in incumbent],
        candidate_seconds=list(candidate_seconds or [1.0] * n),
        incumbent_seconds=list(incumbent_seconds or [1.0] * n),
    )


def test_known_answer_auroc_ap_and_grid_library_lock():
    labels = [False, False, True, True]
    separating = epb.score_block(
        [0.1, 0.2, 0.8, 0.9], labels, name="separating", seconds_per_record=0.1
    )
    reversed_block = epb.score_block(
        [0.9, 0.8, 0.2, 0.1], labels, name="reversed", seconds_per_record=0.1
    )
    assert separating["auroc"] == 1.0
    assert separating["auprc"] == 1.0
    assert reversed_block["auroc"] == 0.0
    assert reversed_block["auprc"] == pytest.approx(5 / 12)

    grid = np.resize(np.array([0.05, 0.20, 0.35, 0.65, 0.80, 0.95]), 120)
    grid_block = epb.score_block(
        grid, SHARED_LABELS, name="grid", seconds_per_record=1.0
    )
    assert grid_block["auroc"] == metrics.auroc(grid, SHARED_LABELS)


def test_singleton_cluster_bootstrap_is_bit_exact_with_deployed_row_sibling():
    """The clustered estimator degenerates exactly to singleton row draws."""
    ours = epb.paired_bootstrap_delta_auroc(
        SHARED_LABELS,
        SHARED_CANDIDATE,
        SHARED_INCUMBENT,
        cluster_ids=SHARED_CLUSTER_IDS,
        cluster_field="fixture_cluster",
        n_boot=2000,
        seed=0,
    )
    sibling = sibling_paired_bootstrap_delta(
        SHARED_LABELS.astype(int),
        SHARED_CANDIDATE,
        SHARED_INCUMBENT,
        seed=0,
        n_boot=2000,
    )
    assert ours["ci95_low"] == sibling["ci95_low"]
    assert ours["ci95_high"] == sibling["ci95_high"]
    assert ours["n_valid_resamples"] == sibling["n_valid_resamples"]
    assert set(ours) == {
        "delta_auroc",
        "ci95_low",
        "ci95_high",
        "p_delta_gt_0",
        "n_valid_resamples",
        "n_bootstrap",
        "seed",
        "resampling_unit",
        "cluster_field",
        "n_clusters",
        "max_cluster_multiplicity",
    }


def test_bootstrap_seed_determinism_sensitivity_and_ci_bracket():
    first = epb.paired_bootstrap_delta_auroc(
        SHARED_LABELS,
        SHARED_CANDIDATE,
        SHARED_INCUMBENT,
        cluster_ids=SHARED_CLUSTER_IDS,
        cluster_field="fixture_cluster",
        n_boot=400,
        seed=0,
    )
    repeated = epb.paired_bootstrap_delta_auroc(
        SHARED_LABELS,
        SHARED_CANDIDATE,
        SHARED_INCUMBENT,
        cluster_ids=SHARED_CLUSTER_IDS,
        cluster_field="fixture_cluster",
        n_boot=400,
        seed=0,
    )
    other_seed = epb.paired_bootstrap_delta_auroc(
        SHARED_LABELS,
        SHARED_CANDIDATE,
        SHARED_INCUMBENT,
        cluster_ids=SHARED_CLUSTER_IDS,
        cluster_field="fixture_cluster",
        n_boot=400,
        seed=1,
    )
    assert first == repeated
    assert first["ci95_low"] != other_seed["ci95_low"]
    assert first["ci95_low"] <= first["delta_auroc"] <= first["ci95_high"]


def test_ap_is_structurally_paired_with_distinct_count():
    grid = np.resize(np.array([0.05, 0.20, 0.35, 0.65, 0.80, 0.95]), 120)
    continuous = np.linspace(0.001, 0.999, 120)
    grid_block = epb.score_block(
        grid, SHARED_LABELS, name="grid", seconds_per_record=1.0
    )
    continuous_block = epb.score_block(
        continuous, SHARED_LABELS, name="continuous", seconds_per_record=0.25
    )
    assert set(grid_block) >= {"auprc", "distinct_scores"}
    assert set(continuous_block) >= {"auprc", "distinct_scores"}
    assert grid_block["distinct_scores"] <= 6
    assert continuous_block["distinct_scores"] == 120
    assert continuous_block["distinct_scores"] > 10 * grid_block["distinct_scores"]


def test_auprc_has_one_ast_call_site_and_one_dict_literal():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))

    def is_auprc_call(node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        if isinstance(node.func, ast.Name):
            return node.func.id == "auprc"
        return (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "metrics"
            and node.func.attr == "auprc"
        )

    calls = [node for node in ast.walk(tree) if is_auprc_call(node)]
    assert len(calls) == 1
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_ap_with_distinct"
    )
    assert calls[0] in list(ast.walk(helper))
    helper_return = next(
        node for node in ast.walk(helper) if isinstance(node, ast.Return)
    )
    assert isinstance(helper_return.value, ast.Dict)
    literal_keys = {
        key.value
        for key in helper_return.value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert literal_keys == {"auprc", "distinct_scores"}


def test_out_of_range_scores_refuse_calibration_but_keep_rank_metrics(monkeypatch):
    def forbidden_ece(*args, **kwargs):
        pytest.fail("score_block called ece on scores outside the unit interval")

    monkeypatch.setattr(epb, "ece", forbidden_ece)
    block = epb.score_block(
        [-4.2, 7.1, -1.0, 0.5],
        [False, True, False, True],
        name="raw_logits",
        seconds_per_record=0.2,
    )
    assert block["in_unit_interval"] is False
    assert block["ece"] is None
    assert block["brier"] is None
    assert block["reliability"] is None
    assert block["resolution"] is None
    assert block["reliability_bins"] is None
    assert np.isfinite(block["auroc"])
    assert np.isfinite(block["auprc"])
    assert metrics.ece([(5.0, True), (-3.0, False)]) == 0.0


def test_score_block_refuses_single_class():
    with pytest.raises(ValueError, match=r"n_pos=4 n_neg=0"):
        epb.score_block(
            [0.1, 0.2, 0.3, 0.4],
            [True, True, True, True],
            name="all_true",
            seconds_per_record=None,
        )
    with pytest.raises(ValueError, match=r"n_pos=0 n_neg=4"):
        epb.score_block(
            [0.1, 0.2, 0.3, 0.4],
            [False, False, False, False],
            name="all_false",
            seconds_per_record=None,
        )


def _repeated_hash_gold() -> list[dict]:
    return [
        {
            "source_hash": 99,
            "tag": "correct",
            "verdict": "correct",
            "confidence": "high",
            "call_log": [
                {"duration_s": 1.0},
                {"duration_s": 0.25},
                {"duration_s": None},
            ],
        },
        {
            "source_hash": 99,
            "tag": "wrong_relation",
            "verdict": "incorrect",
            "confidence": "low",
            "call_log": [{"duration_s": 2.0}],
        },
    ]


def _paired_score_rows() -> list[dict]:
    return [
        {"record_id": "1:99", "candidate_score": 0.2, "candidate_seconds": 0.2},
        {"record_id": "0:99", "candidate_score": 0.9, "candidate_seconds": 0.1},
    ]


def test_pairing_is_bijective_and_ordinal_despite_repeated_source_hash(tmp_path):
    gold_path = tmp_path / "gold.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    _write_jsonl(gold_path, _repeated_hash_gold())
    _write_jsonl(scores_path, _paired_score_rows())

    pairs = epb.load_pairs(scores_path, gold_path)
    assert pairs.record_ids == ["0:99", "1:99"]
    assert pairs.cluster_ids == ["99", "99"]
    assert pairs.cluster_field == "source_hash"
    assert pairs.labels == [True, False]
    assert pairs.candidate == [0.9, 0.2]
    assert pairs.incumbent == [0.95, 0.35]
    assert pairs.candidate_seconds == [0.1, 0.2]
    assert pairs.incumbent_seconds == [1.25, 2.0]


def test_pairing_refuses_missing_extra_and_duplicate_ids(tmp_path):
    gold_path = tmp_path / "gold.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    _write_jsonl(gold_path, _repeated_hash_gold())

    valid = list(reversed(_paired_score_rows()))
    _write_jsonl(scores_path, valid[:1])
    with pytest.raises(
        epb.PairingError, match=r"gold=2 scores=1 missing=1.*length_mismatch=1"
    ):
        epb.load_pairs(scores_path, gold_path)

    _write_jsonl(
        scores_path,
        valid + [{"record_id": "2:99", "candidate_score": 0.5}],
    )
    with pytest.raises(
        epb.PairingError, match=r"gold=2 scores=3 missing=0 extra=1.*length_mismatch=1"
    ):
        epb.load_pairs(scores_path, gold_path)

    _write_jsonl(scores_path, [valid[0], valid[0], valid[1]])
    with pytest.raises(
        epb.PairingError, match=r"gold=2 scores=3 missing=0 extra=0 duplicate=1"
    ):
        epb.load_pairs(scores_path, gold_path)


def test_pairing_refuses_off_grid_incumbent(tmp_path):
    gold_path = tmp_path / "gold.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    _write_jsonl(
        gold_path,
        [
            {
                "source_hash": 7,
                "tag": "correct",
                "verdict": "correct",
                "confidence": "certain",
            }
        ],
    )
    _write_jsonl(scores_path, [{"record_id": "0:7", "candidate_score": 0.5}])
    with pytest.raises(ValueError, match="grid_score returned None"):
        epb.load_pairs(scores_path, gold_path)


def test_self_mode_loads_d1_holdout_scores_shape(tmp_path):
    scores_path = tmp_path / "holdout_scores.jsonl"
    rows = [
        {
            "row_index": 0,
            "source_hash": 44,
            "gold_correct": True,
            "battery_score": 0.91,
            "incumbent_score": 0.80,
            "elapsed_s_battery": 0.0,
            "elapsed_s_incumbent": 4.0,
        },
        {
            "row_index": 1,
            "source_hash": 44,
            "gold_correct": False,
            "battery_score": 0.12,
            "incumbent_score": 0.20,
            "elapsed_s_battery": 0.25,
            "elapsed_s_incumbent": 3.5,
        },
    ]
    _write_jsonl(scores_path, rows)
    pairs = epb.load_pairs(scores_path)
    assert pairs.record_ids == ["0:44", "1:44"]
    assert pairs.cluster_ids == ["44", "44"]
    assert pairs.cluster_field == "source_hash"
    assert pairs.labels == [True, False]
    assert pairs.candidate == [0.91, 0.12]
    assert pairs.incumbent == [0.8, 0.2]
    assert pairs.candidate_seconds == [0.0, 0.25]
    assert pairs.incumbent_seconds == [4.0, 3.5]


def test_gate_reads_only_delta_auroc_ci_not_cost_or_distinct_count():
    n = 500
    labels = [True] * 250 + [False] * 250
    candidate = np.arange(n) / (n - 1)
    positive_grid = [0.65, 0.80, 0.95]
    negative_grid = [0.05, 0.20, 0.35]
    incumbent = [
        positive_grid[i % 3] if label else negative_grid[i % 3]
        for i, label in enumerate(labels)
    ]
    pairs = _pairs(
        labels,
        candidate,
        incumbent,
        candidate_seconds=[0.25] * n,
        incumbent_seconds=[1.0] * n,
    )
    decision = epb.evaluate(pairs, n_boot=200, seed=0)
    assert decision["candidate"]["distinct_scores"] == 500
    # Renamed from `speedup_x`: the bare name read as a cost win on an ADDITIVE
    # arm, where you pay the incumbent AND the candidate. Both semantics now ship.
    assert decision["cost"]["speedup_x_if_replacement"] == 4.0
    assert decision["cost"]["cost_ratio_vs_incumbent_if_additive"] == pytest.approx(1.25)
    assert decision["paired_bootstrap"]["ci95_low"] < 0.0
    assert decision["gate"]["passed"] is False
    assert decision["verdict"] == "NO-GO"


def test_evaluate_refuses_single_class_before_nan_can_escape():
    pairs = _pairs(
        [True, True, True, True],
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.5],
    )
    with pytest.raises(ValueError, match=r"n_pos=4 n_neg=0"):
        epb.evaluate(pairs, n_boot=20, seed=0)


def _self_rows(n: int = 12) -> list[dict]:
    return [
        {
            "row_index": i,
            "source_hash": 1000 + i,
            "gold_correct": i % 2 == 0,
            "battery_score": (i + 1) / (n + 1),
            "incumbent_score": 0.8 if i % 2 == 0 else 0.2,
            "elapsed_s_battery": 0.1,
            "elapsed_s_incumbent": 0.4,
        }
        for i in range(n)
    ]


def test_self_mode_refuses_missing_source_hash_even_with_record_id(tmp_path):
    scores_path = tmp_path / "holdout_scores.jsonl"
    rows = _self_rows()
    rows[0]["record_id"] = "explicit-record-id"
    rows[0].pop("source_hash")
    _write_jsonl(scores_path, rows)

    with pytest.raises(epb.PairingError, match=r"missing required field source_hash"):
        epb.load_pairs(scores_path)


def test_absent_candidate_timing_is_null_never_zero(tmp_path):
    scores_path = tmp_path / "holdout_scores.jsonl"

    absent_rows = _self_rows()
    for row in absent_rows:
        row.pop("elapsed_s_battery")
    _write_jsonl(scores_path, absent_rows)
    absent_pairs = epb.load_pairs(scores_path)
    assert absent_pairs.candidate_seconds == [None] * len(absent_rows)
    absent_decision = epb.evaluate(absent_pairs, n_boot=64, seed=7)
    assert absent_decision["candidate"]["seconds_per_record"] is None
    assert absent_decision["cost"]["candidate_s_per_record"] is None
    assert absent_decision["cost"]["speedup_x_if_replacement"] is None
    assert absent_decision["cost"]["candidate_records_timed"] == 0

    zero_rows = _self_rows()
    for row in zero_rows:
        row["elapsed_s_battery"] = 0.0
    _write_jsonl(scores_path, zero_rows)
    zero_pairs = epb.load_pairs(scores_path)
    assert zero_pairs.candidate_seconds == [0.0] * len(zero_rows)
    assert all(value is not None for value in zero_pairs.candidate_seconds)
    zero_decision = epb.evaluate(zero_pairs, n_boot=64, seed=7)
    assert zero_decision["candidate"]["seconds_per_record"] == 0.0
    assert zero_decision["cost"]["candidate_s_per_record"] == 0.0
    assert zero_decision["cost"]["candidate_records_timed"] == len(zero_rows)

    mixed_rows = _self_rows()
    for row in mixed_rows:
        row.pop("elapsed_s_battery")
    mixed_rows[0]["elapsed_s_battery"] = 0.25
    _write_jsonl(scores_path, mixed_rows)
    mixed_pairs = epb.load_pairs(scores_path)
    assert mixed_pairs.candidate_seconds == [0.25] + [None] * (
        len(mixed_rows) - 1
    )
    mixed_decision = epb.evaluate(mixed_pairs, n_boot=64, seed=7)
    assert mixed_decision["candidate"]["seconds_per_record"] is None
    assert mixed_decision["cost"]["candidate_s_per_record"] is None
    assert mixed_decision["cost"]["candidate_records_timed"] == 1
    assert mixed_decision["cost"]["speedup_x_if_replacement"] is None


def test_incumbent_timing_absent_when_call_log_carries_no_duration(tmp_path):
    gold_path = tmp_path / "gold.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    gold_rows = [
        {
            "source_hash": 3000,
            "tag": "correct",
            "verdict": "correct",
            "confidence": "high",
            "call_log": [],
        },
        {
            "source_hash": 3001,
            "tag": "wrong_relation",
            "verdict": "incorrect",
            "confidence": "low",
            "call_log": [{}],
        },
        {
            "source_hash": 3002,
            "tag": "correct",
            "verdict": "correct",
            "confidence": "medium",
            "call_log": [
                {"duration_s": 1.0},
                {"duration_s": 0.25},
                {"duration_s": None},
            ],
        },
    ]
    score_rows = [
        {
            "record_id": f"{i}:{row['source_hash']}",
            "candidate_score": [0.9, 0.1, 0.8][i],
            "candidate_seconds": 0.1,
        }
        for i, row in enumerate(gold_rows)
    ]
    _write_jsonl(gold_path, gold_rows)
    _write_jsonl(scores_path, score_rows)

    pairs = epb.load_pairs(scores_path, gold_path)
    assert pairs.incumbent_seconds == [None, None, 1.25]
    assert pairs.incumbent_untimed_calls == 2
    decision = epb.evaluate(pairs, n_boot=64, seed=7)
    assert decision["cost"]["incumbent_s_per_record"] is None
    assert decision["cost"]["incumbent_records_timed"] == 1
    assert decision["cost"]["incumbent_untimed_calls"] == 2
    assert decision["cost"]["speedup_x_if_replacement"] is None


def test_pair_mode_reproduce_authenticates_scores_and_gold(tmp_path, capsys):
    gold_path = tmp_path / "gold.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    decision_path = tmp_path / "decision.json"
    gold_rows = [
        {
            "source_hash": 2000 + i,
            "tag": "correct" if i % 2 == 0 else "wrong_relation",
            "verdict": "correct" if i % 2 == 0 else "incorrect",
            "confidence": "medium",
            "call_log": [{"duration_s": 0.4}],
        }
        for i in range(12)
    ]
    score_rows = [
        {
            "record_id": f"{i}:{row['source_hash']}",
            "candidate_score": (i + 1) / 13,
            "candidate_seconds": 0.1,
        }
        for i, row in enumerate(gold_rows)
    ]
    _write_jsonl(gold_path, gold_rows)
    _write_jsonl(scores_path, score_rows)
    decision = epb.evaluate(
        epb.load_pairs(scores_path, gold_path), n_boot=64, seed=7
    )
    epb.write_decision(
        decision_path,
        decision,
        scores_path=scores_path,
        gold_path=gold_path,
    )
    assert epb.reproduce(decision_path) == 0
    capsys.readouterr()

    gold_path.write_text(
        gold_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    assert epb.reproduce(decision_path) == 1
    output = capsys.readouterr()
    assert "gold.sha256" in output.err


def test_write_reproduce_round_trip_mutation_digest_and_unknown_keys(
    tmp_path, capsys
):
    scores_path = tmp_path / "holdout_scores.jsonl"
    decision_path = tmp_path / "new" / "directory" / "decision.json"
    _write_jsonl(scores_path, _self_rows())
    decision = epb.evaluate(epb.load_pairs(scores_path), n_boot=64, seed=7)
    epb.write_decision(decision_path, decision, scores_path=scores_path)
    assert decision_path.read_text(encoding="utf-8").endswith("\n")

    assert epb.reproduce(decision_path) == 0
    faithful_output = capsys.readouterr()
    match = re.search(r"compared_fields=(\d+)", faithful_output.out)
    assert match is not None and int(match.group(1)) > 0

    artifact = json.loads(decision_path.read_text(encoding="utf-8"))
    artifact.pop("inputs")
    artifact.update(
        {
            "decision_rule": "metadata owned by D1",
            "verdict_reason": "metadata owned by D1",
            "splits": {"fit": {}, "test": {}},
            "split_disjointness": {"asserted": True},
            "join": {"n_joined": 12},
            "frozen_combiner": {"sha256": "metadata"},
            "d1_sensitivity": {"ci95_low": -0.5},
            "killgate": {"passed": True},
            "scores_file": {
                "path": str(scores_path),
                "sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest(),
                "n_rows": 12,
            },
            "reproduce": {"tolerance": 1e-9},
        }
    )
    decision_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert epb.reproduce(decision_path) == 0
    extra_output = capsys.readouterr()
    match = re.search(r"compared_fields=(\d+)", extra_output.out)
    assert match is not None and int(match.group(1)) > 0

    undeclared = json.loads(json.dumps(artifact))
    undeclared["paired_bootstrap"].pop("resampling_unit")
    decision_path.write_text(
        json.dumps(undeclared, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert epb.reproduce(decision_path) == 1
    undeclared_output = capsys.readouterr()
    assert "does not declare its resampling unit" in undeclared_output.err

    stale = json.loads(json.dumps(artifact))
    stale["paired_bootstrap"].pop("n_clusters")
    decision_path.write_text(
        json.dumps(stale, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert epb.reproduce(decision_path) == 1
    stale_output = capsys.readouterr()
    assert "paired_bootstrap.n_clusters: recorded=<missing>" in stale_output.err

    mutated = json.loads(json.dumps(artifact))
    mutated["candidate"]["auroc"] += 0.125
    decision_path.write_text(
        json.dumps(mutated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert epb.reproduce(decision_path) == 1
    mutation_output = capsys.readouterr()
    assert "candidate.auroc" in mutation_output.err

    decision_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    score_lines = scores_path.read_text(encoding="utf-8").splitlines()
    scores_path.write_text("\n".join(score_lines[:-1]) + "\n", encoding="utf-8")
    assert epb.reproduce(decision_path) == 1
    digest_output = capsys.readouterr()
    assert "scores_file.sha256" in digest_output.err
    assert "recorded=" in digest_output.err and "recomputed=" in digest_output.err


def test_module_documents_measured_contract_and_has_no_numeric_doc_anchors():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "compute_deployed_baseline_replication.py" in source
    assert "compute_belief_model_ladder.py" in source
    assert "test_auprc_is_order_invariant" in source
    assert "test_auprc_matches_sklearn_average_precision" in source
    assert "This node deliberately does not edit those two scripts" in source
    assert "SIX distinct values" in source
    assert "0.05x63" in source and "0.95x168" in source
    assert re.search(r"[a-z_]+\.py:[0-9]+", source) is None

    tree = ast.parse(source)
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots.isdisjoint({"sklearn", "scipy", "mlx"})


def test_evaluator_has_no_decorative_probe_combiner_import():
    assert "probe_combiner" not in SCRIPT.read_text(encoding="utf-8")


@pytest.mark.skipif(not REAL_GOLD.exists(), reason="gitignored holdout_cc file is absent")
def test_real_holdout_smoke_uses_six_grid_cells_and_artifact_cost(tmp_path):
    gold_rows = [
        json.loads(line)
        for line in REAL_GOLD.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scores_path = tmp_path / "scores.jsonl"
    _write_jsonl(
        scores_path,
        [
            {
                "record_id": f"{i}:{row['source_hash']}",
                "candidate_score": 0.5,
            }
            for i, row in enumerate(gold_rows)
        ],
    )
    pairs = epb.load_pairs(scores_path, REAL_GOLD)
    expected_seconds = sum(
        sum(call.get("duration_s") or 0.0 for call in (row.get("call_log") or []))
        for row in gold_rows
    ) / len(gold_rows)
    incumbent_s = sum(pairs.incumbent_seconds) / len(pairs.incumbent_seconds)
    block = epb.score_block(
        pairs.incumbent,
        pairs.labels,
        name="incumbent",
        seconds_per_record=incumbent_s,
    )
    assert block["n"] == 500
    assert block["distinct_scores"] == 6
    assert block["seconds_per_record"] == pytest.approx(expected_seconds, abs=1e-12)


def test_clustered_bootstrap_ci_is_wider_than_singleton_clusters():
    labels: list[bool] = []
    candidate: list[float] = []
    incumbent: list[float] = []
    cluster_ids: list[str] = []
    for cluster_index in range(20):
        candidate_is_correct = cluster_index % 2 == 0
        for row_index in range(10):
            is_positive = row_index < 5
            labels.append(is_positive)
            candidate.append(
                0.9 if is_positive == candidate_is_correct else 0.1
            )
            incumbent.append(0.6 if is_positive else 0.4)
            cluster_ids.append(f"source-{cluster_index:02d}")

    clustered = epb.paired_bootstrap_delta_auroc(
        labels,
        candidate,
        incumbent,
        cluster_ids=cluster_ids,
        cluster_field="source_hash",
        n_boot=2000,
        seed=0,
    )
    singleton = epb.paired_bootstrap_delta_auroc(
        labels,
        candidate,
        incumbent,
        cluster_ids=[f"{i:04d}" for i in range(len(labels))],
        cluster_field="fixture_row",
        n_boot=2000,
        seed=0,
    )

    clustered_width = clustered["ci95_high"] - clustered["ci95_low"]
    singleton_width = singleton["ci95_high"] - singleton["ci95_low"]
    assert clustered_width > singleton_width
    assert clustered["n_valid_resamples"] == 2000
    assert singleton["n_valid_resamples"] == 2000
