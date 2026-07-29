from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from indra.statements import Activation, Agent, Evidence
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import score_current_indra_counts_hybrid_readers as adapter  # noqa: E402


def _panel() -> tuple[list[Activation], list[list[Evidence]], list[dict], list[dict]]:
    statements: list[Activation] = []
    extras: list[list[Evidence]] = []
    targets: list[dict] = []
    gold: list[dict] = []
    for index in range(20):
        statement = Activation(
            Agent(f"A{index}"),
            Agent(f"B{index}"),
            evidence=[
                Evidence(
                    source_api="reach",
                    pmid=str(1000 + index),
                    text=f"Promoter A{index} activates B{index}.",
                ),
                Evidence(
                    source_api="signor",
                    pmid=str(2000 + index),
                    text=f"A{index} activates B{index}.",
                ),
            ],
        )
        hidden_support = Activation(
            Agent(f"A{index}"),
            Agent(f"B{index}"),
            evidence=[Evidence(source_api="signor", text="Database-only support.")],
        )
        statement.supports = [hidden_support]
        statement.supported_by = [hidden_support]
        statement.belief = 0.01 if index % 2 else 0.99
        statements.append(statement)
        extras.append(
            [
                Evidence(
                    source_api="sparser",
                    pmid=str(3000 + index),
                    text=f"A{index} induces B{index}.",
                ),
                Evidence(
                    source_api="signor",
                    pmid=str(4000 + index),
                    text=f"Database support for A{index} and B{index}.",
                ),
            ]
        )
        targets.append({"reader_eligible": True, "statement_id": statement.uuid})
        gold.append(
            {
                "fold_id": index % 10,
                "label": index // 10,
                "statement_id": statement.uuid,
            }
        )
    return statements, extras, targets, gold


def _tiny_estimator(fold_id: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=8,
        max_depth=3,
        max_features="sqrt",
        random_state=400 + fold_id,
        n_jobs=1,
    )


def test_reader_feature_contract_is_exact_current_counts_order() -> None:
    assert adapter.PANEL_ID == "readers_only_1676"
    assert adapter.READER_SOURCES == (
        "reach",
        "sparser",
        "medscan",
        "rlimsp",
        "trips",
    )
    assert (
        adapter.RF_N_ESTIMATORS,
        adapter.RF_MAX_DEPTH,
        adapter.RF_RANDOM_STATE,
    ) == (2000, 13, 4)
    source = adapter._feature_contract(adapter.SOURCE_CONFIGURATION)
    full = adapter._feature_contract(adapter.FULL_CONFIGURATION)
    assert source["feature_count"] == 5
    assert source["feature_names"] == [
        f"direct_source_count:{name}" for name in adapter.READER_SOURCES
    ]
    assert full["feature_count"] == 65
    assert full["feature_names"][:5] == source["feature_names"]
    assert full["feature_names"][5:10] == [
        f"more_specific_source_count:{name}" for name in adapter.READER_SOURCES
    ]
    assert full["feature_names"][-6:] == [
        "has_residue_and_position",
        "statement_member_count",
        "direct_unique_pmid_count",
        "more_specific_unique_pmid_count",
        "direct_promoter_sentence_fraction",
        "average_direct_evidence_sentence_length",
    ]
    assert full["feature_names_sha256"] == hashlib.sha256(
        adapter._canonical_bytes(full["feature_names"])
    ).hexdigest()
    assert full["source_list_order"] == list(adapter.READER_SOURCES)
    assert full["non_source_features_are_reader_projected"] is True
    assert source["feature_names_sha256"] == adapter.PINNED_SOURCE_FEATURE_SHA256
    assert full["feature_names_sha256"] == adapter.PINNED_FULL_FEATURE_SHA256
    assert set(adapter._verify_literal_contract()) == {
        adapter.SOURCE_CONFIGURATION.config_id,
        adapter.FULL_CONFIGURATION.config_id,
    }


@pytest.mark.parametrize("field", ["panel", "sources", "rf"])
def test_literal_contract_rejects_runtime_drift(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    if field == "panel":
        monkeypatch.setattr(adapter, "PANEL_ID", "reader_alias")
    elif field == "sources":
        monkeypatch.setattr(adapter, "READER_SOURCES", tuple(reversed(adapter.READER_SOURCES)))
    else:
        monkeypatch.setattr(adapter, "RF_RANDOM_STATE", 5)
    with pytest.raises(adapter.ContractError, match="drifted"):
        adapter._verify_literal_contract()


def test_exact_rf_contract_is_enforced() -> None:
    adapter._assert_rf_estimator_contract(adapter._new_rf(0, n_jobs=1))
    with pytest.raises(adapter.ContractError, match="RF fit estimator contract drifted"):
        adapter._assert_rf_estimator_contract(_tiny_estimator(0))


def test_reader_projection_removes_database_evidence_and_graph_links() -> None:
    statements, extras, targets, gold = _panel()
    projected, projected_extra, positions, audits = adapter._project_reader_inputs(
        statements, extras, targets, gold
    )
    assert positions == list(range(20))
    assert [statement.uuid for statement in projected] == [
        row["statement_id"] for row in gold
    ]
    assert all([evidence.source_api for evidence in statement.evidence] == ["reach"] for statement in projected)
    assert all([evidence.source_api for evidence in row] == ["sparser"] for row in projected_extra)
    assert all(not statement.supports and not statement.supported_by for statement in projected)
    assert all(statement.belief is None for statement in projected)
    assert all(row["removed_direct_evidence_count"] == 1 for row in audits)
    assert all(row["removed_inherited_evidence_count"] == 1 for row in audits)
    assert all(row["stored_statement_belief_cleared"] is True for row in audits)
    assert all(set(row["projected_direct_source_counts"]) == {"reach"} for row in audits)
    assert all(set(row["projected_inherited_source_counts"]) == {"sparser"} for row in audits)
    # The source objects remain untouched; the adapter operates on copies.
    assert all(len(statement.evidence) == 2 for statement in statements)
    assert all(statement.supports and statement.supported_by for statement in statements)
    assert all(statement.belief in (0.01, 0.99) for statement in statements)


def test_cross_fit_defaults_to_exact_frozen_reader_panel() -> None:
    statements, extras, targets, gold = _panel()
    projected, projected_extra, _, audits = adapter._project_reader_inputs(
        statements, extras, targets, gold
    )
    with pytest.raises(adapter.ContractError, match="reader panel row/order"):
        adapter._cross_fit(
            projected,
            projected_extra,
            gold,
            estimator_factory=_tiny_estimator,
            projection_audits=audits,
        )


def test_reader_cross_fit_is_ordered_label_isolated_and_hybrid_equivalent() -> None:
    statements, extras, targets, gold = _panel()
    projected, projected_extra, _, audits = adapter._project_reader_inputs(
        statements, extras, targets, gold
    )
    predictions, provenance, fits, hybrid_audit = adapter._cross_fit(
        projected,
        projected_extra,
        gold,
        estimator_factory=_tiny_estimator,
        projection_audits=audits,
        enforce_frozen_contract=False,
    )
    expected_ids = [row["statement_id"] for row in gold]
    assert set(predictions) == set(adapter.PREDICTION_FILENAMES)
    assert all(
        [row["statement_id"] for row in rows] == expected_ids
        for rows in predictions.values()
    )
    assert all(
        set(row) == {"statement_id", "probability_correct"}
        for rows in predictions.values()
        for row in rows
    )
    assert len(provenance) == 60
    assert len(fits) == 20
    assert all(row["panel_id"] == adapter.PANEL_ID for row in fits)
    assert all(row["train_statement_count"] == 18 for row in fits)
    assert all(row["test_statement_count"] == 2 for row in fits)
    assert all(row["train_test_statement_intersection"] == 0 for row in fits)
    assert all(row["test_label_count_not_passed_to_fit"] == 2 for row in fits)
    assert all(row["outside_reader_statement_labels_available_to_adapter"] is False for row in fits)
    assert all(len(row["fitted_state_sha256"]) == 64 for row in fits)
    assert all(row["evaluation_design"] == adapter.EVALUATION_DESIGN_LABEL for row in fits)

    counts = predictions[adapter.ARM_COUNTS_FULL]
    hybrid = predictions[adapter.ARM_HYBRID_FULL]
    differences = [
        abs(left["probability_correct"] - right["probability_correct"])
        for left, right in zip(counts, hybrid, strict=True)
    ]
    assert max(differences) <= 4 * np.finfo(float).eps
    assert hybrid_audit["hybrid_simple_fallback_evidence_entries"] == 0
    assert hybrid_audit["rows_compared"] == 20
    assert hybrid_audit["exact_equal_probability_rows"] + hybrid_audit[
        "unequal_probability_rows"
    ] == 20
    assert hybrid_audit["numerically_equivalent_within_tolerance"] is True
    assert hybrid_audit["pareto_point_policy"] == (
        "alias_of_counts_component_not_a_distinct_model_point"
    )

    # Fold-0 labels are unavailable to fold-0 fits.  Changing only those labels
    # cannot change any fold-0 prediction (but may change other folds' fits).
    changed_gold = copy.deepcopy(gold)
    for row in changed_gold:
        if row["fold_id"] == 0:
            row["label"] = 1 - row["label"]
    changed, _, _, _ = adapter._cross_fit(
        projected,
        projected_extra,
        changed_gold,
        estimator_factory=_tiny_estimator,
        projection_audits=audits,
        enforce_frozen_contract=False,
    )
    fold_zero_ids = {row["statement_id"] for row in gold if row["fold_id"] == 0}
    for arm_id in adapter.PREDICTION_FILENAMES:
        original_map = {
            row["statement_id"]: row["probability_correct"]
            for row in predictions[arm_id]
        }
        changed_map = {
            row["statement_id"]: row["probability_correct"]
            for row in changed[arm_id]
        }
        assert {item: original_map[item] for item in fold_zero_ids} == {
            item: changed_map[item] for item in fold_zero_ids
        }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("direct_source", "direct evidence outside reader sources"),
        ("extra_source", "extra evidence outside reader sources"),
        ("supports", "stale supports/supported_by"),
        ("supported_by", "stale supports/supported_by"),
        ("belief", "stale stored statement belief"),
    ],
)
def test_cross_fit_rejects_stale_projection_before_estimator_creation(
    mutation: str, message: str
) -> None:
    statements, extras, targets, gold = _panel()
    projected, projected_extra, _, audits = adapter._project_reader_inputs(
        statements, extras, targets, gold
    )
    if mutation == "direct_source":
        projected[0].evidence.append(Evidence(source_api="signor", text="outside"))
    elif mutation == "extra_source":
        projected_extra[0].append(Evidence(source_api="signor", text="outside"))
    elif mutation == "supports":
        projected[0].supports = [projected[1]]
    elif mutation == "supported_by":
        projected[0].supported_by = [projected[1]]
    else:
        projected[0].belief = 0.5
    estimator_created = False

    def forbidden_estimator(_fold_id: int) -> RandomForestClassifier:
        nonlocal estimator_created
        estimator_created = True
        raise AssertionError("estimator factory must not be reached")

    with pytest.raises(adapter.ContractError, match=message):
        adapter._cross_fit(
            projected,
            projected_extra,
            gold,
            estimator_factory=forbidden_estimator,
            projection_audits=audits,
            enforce_frozen_contract=False,
        )
    assert estimator_created is False


def test_reader_adapter_has_no_all_source_gold_input() -> None:
    parameters = inspect.signature(adapter.materialize).parameters
    assert "reader_gold_path" in parameters
    assert "all_gold_path" not in parameters
    assert "evidence_adjudication_path" not in parameters
    actions = adapter._parser()._actions
    option_strings = {option for action in actions for option in action.option_strings}
    assert "--reader-gold-path" in option_strings
    assert "--all-gold-path" not in option_strings


def test_registry_contract_binds_all_three_runtime_classes(tmp_path: Path) -> None:
    registry_path = ROOT / "data/comparison/scorers.json"
    descriptor, _, bindings, runtime = adapter._registry_contract(registry_path)
    assert descriptor["sha256"] == adapter.PINNED_REGISTRY_SHA256
    assert descriptor["verification"] == "exact_sha256_regular_file_no_symlink"
    assert set(bindings) == {
        "indra_1.24.0_simple_default",
        "indra_1.24.0_counts_unfitted",
        "indra_1.24.0_hybrid_unfitted",
    }
    assert all(len(row["registry_entry_sha256"]) == 64 for row in bindings.values())
    assert runtime["belief_init"]["sha256"] == adapter.PINNED_BELIEF_INIT_SHA256
    assert runtime["belief_skl"]["sha256"] == adapter.PINNED_BELIEF_SKL_SHA256
    assert runtime["default_prior_resource"]["sha256"] == (
        adapter.PINNED_DEFAULT_PRIOR_SHA256
    )

    forged = json.loads(registry_path.read_text())
    counts = next(
        row for row in forged["scorers"] if row["scorer_id"] == "indra_1.24.0_counts_unfitted"
    )
    counts["implementation_sha256"] = "0" * 64
    forged_path = tmp_path / "forged_registry.json"
    forged_path.write_text(json.dumps(forged, sort_keys=True) + "\n")
    with pytest.raises(adapter.ContractError, match="SHA-256 mismatch"):
        adapter._registry_contract(forged_path)


def test_registry_rejects_self_consistent_dual_hash_mutation_even_if_repinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path = ROOT / "data/comparison/scorers.json"
    forged = json.loads(registry_path.read_text())
    forged["installed_runtime"]["belief_skl_sha256"] = "0" * 64
    for scorer in forged["scorers"]:
        if scorer.get("scorer_id") in {
            "indra_1.24.0_counts_unfitted",
            "indra_1.24.0_hybrid_unfitted",
        }:
            scorer["implementation_sha256"] = "0" * 64
    forged_path = tmp_path / "dual_mutation_registry.json"
    forged_path.write_text(json.dumps(forged, sort_keys=True) + "\n")
    monkeypatch.setattr(
        adapter,
        "PINNED_REGISTRY_SHA256",
        hashlib.sha256(forged_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(adapter.ContractError, match="installed_runtime"):
        adapter._registry_contract(forged_path)


def test_registry_rejects_class_mutation_even_if_file_pin_is_repinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path = ROOT / "data/comparison/scorers.json"
    forged = json.loads(registry_path.read_text())
    counts = next(
        row for row in forged["scorers"] if row["scorer_id"] == "indra_1.24.0_counts_unfitted"
    )
    counts["class"] = "indra.belief.skl.HybridScorer"
    forged_path = tmp_path / "class_mutation_registry.json"
    forged_path.write_text(json.dumps(forged, sort_keys=True) + "\n")
    monkeypatch.setattr(
        adapter,
        "PINNED_REGISTRY_SHA256",
        hashlib.sha256(forged_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(adapter.ContractError, match="exact class identity"):
        adapter._registry_contract(forged_path)


def test_dependency_sources_are_pinned_before_import_and_reject_drift(
    tmp_path: Path,
) -> None:
    captures = adapter._preverify_dependency_sources(ROOT / "scripts")
    assert set(captures) == set(adapter.DEPENDENCY_SOURCE_PINS)
    for role, capture in captures.items():
        assert capture.sha256 == adapter.DEPENDENCY_SOURCE_PINS[role][2]

    copied = tmp_path / "scripts"
    copied.mkdir()
    for _, filename, _ in adapter.DEPENDENCY_SOURCE_PINS.values():
        shutil.copyfile(ROOT / "scripts" / filename, copied / filename)
    drifted_name = adapter.DEPENDENCY_SOURCE_PINS["bayesian_panel_adapter"][1]
    with (copied / drifted_name).open("ab") as handle:
        handle.write(b"\n# drift\n")
    with pytest.raises(adapter.ContractError, match="SHA-256 mismatch"):
        adapter._preverify_dependency_sources(copied)


def test_preverified_dependencies_import_from_and_revalidate_exact_origins() -> None:
    dependencies = adapter._load_pinned_dependencies()
    assert set(dependencies.descriptors) == set(adapter.DEPENDENCY_SOURCE_PINS)
    modules = {
        "bayesian_panel_adapter": dependencies.bayes,
        "all_source_counts_adapter": dependencies.all_source,
        "hierarchy_graph_adapter": dependencies.hierarchy,
        "identity_adapter": dependencies.base,
    }
    for role, module in modules.items():
        expected_name, filename, expected_sha = adapter.DEPENDENCY_SOURCE_PINS[role]
        assert module.__name__ == expected_name
        assert Path(module.__file__).resolve() == (ROOT / "scripts" / filename).resolve()
        assert dependencies.descriptors[role]["sha256"] == expected_sha


def test_dependency_source_symlink_is_rejected(tmp_path: Path) -> None:
    copied = tmp_path / "scripts"
    copied.mkdir()
    for role, (_, filename, _) in adapter.DEPENDENCY_SOURCE_PINS.items():
        source = ROOT / "scripts" / filename
        destination = copied / filename
        if role == "bayesian_panel_adapter":
            os.symlink(source, destination)
        else:
            shutil.copyfile(source, destination)
    with pytest.raises(adapter.ContractError, match="symbolic links are forbidden"):
        adapter._preverify_dependency_sources(copied)


def test_reader_module_has_no_top_level_local_adapter_imports() -> None:
    source = (ROOT / "scripts/score_current_indra_counts_hybrid_readers.py").read_text()
    tree = __import__("ast").parse(source)
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, __import__("ast").Import)
        for alias in node.names
    }
    assert not imported.intersection(
        module_name for module_name, _, _ in adapter.DEPENDENCY_SOURCE_PINS.values()
    )


def test_comparison_and_release_chains_are_hard_pinned() -> None:
    comparison = ROOT / "data/results/indra_paper_comparison_gold_20260717"
    manifest, targets, gold, target_rows, gold_rows = adapter._comparison_contract(
        comparison / "paper_comparison_gold_manifest.json",
        comparison / "paper_prediction_targets.jsonl",
        comparison / "paper_reader_eligible_released_gold.jsonl",
    )
    assert manifest["sha256"] == adapter.PINNED_COMPARISON_MANIFEST_SHA256
    assert targets["sha256"] == adapter.PINNED_TARGETS_SHA256
    assert gold["sha256"] == adapter.PINNED_READER_GOLD_SHA256
    assert gold["ordered_statement_id_sha256"] == adapter.PINNED_ORDERED_READER_ID_SHA256
    assert len(target_rows) == 1689
    assert len(gold_rows) == 1676
    paper, declared = adapter._paper_manifest_contract(
        ROOT / "data/benchmark/indra_paper_2023.manifest.json"
    )
    assert paper["sha256"] == adapter.PINNED_PAPER_MANIFEST_SHA256
    assert declared["sha256"] == adapter.PINNED_PAPER_PICKLE_SHA256


def test_self_consistent_manifest_and_ledger_mutation_is_rejected(
    tmp_path: Path,
) -> None:
    comparison = ROOT / "data/results/indra_paper_comparison_gold_20260717"
    manifest = json.loads((comparison / "paper_comparison_gold_manifest.json").read_text())
    gold_path = tmp_path / "gold.jsonl"
    mutated = (comparison / "paper_reader_eligible_released_gold.jsonl").read_bytes() + b"\n"
    gold_path.write_bytes(mutated)
    manifest["outputs"]["paper_reader_eligible_released_gold"].update(
        bytes=len(mutated), sha256=hashlib.sha256(mutated).hexdigest()
    )
    manifest_path = tmp_path / "comparison.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    with pytest.raises(adapter.ContractError, match="SHA-256 mismatch"):
        adapter._comparison_contract(
            manifest_path,
            comparison / "paper_prediction_targets.jsonl",
            gold_path,
        )


def test_self_consistent_paper_manifest_and_pickle_mutation_is_rejected(
    tmp_path: Path,
) -> None:
    fake_pickle = tmp_path / "paper.pkl"
    fake_pickle.write_bytes(b"not a release pickle")
    manifest = json.loads(
        (ROOT / "data/benchmark/indra_paper_2023.manifest.json").read_text()
    )
    row = next(
        item for item in manifest["files"] if item["filename"] == "indra_benchmark_corpus.pkl"
    )
    row["bytes"] = fake_pickle.stat().st_size
    row["sha256"] = hashlib.sha256(fake_pickle.read_bytes()).hexdigest()
    manifest_path = tmp_path / "paper_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    with pytest.raises(adapter.ContractError, match="SHA-256 mismatch"):
        adapter._paper_manifest_contract(manifest_path)


def test_graph_fold_dependency_disclosure_is_exact_and_not_inductive() -> None:
    statements = [
        Activation(
            Agent(f"S{index}"),
            Agent(f"O{index}"),
            evidence=[Evidence(source_api="reach", text=f"S{index} activates O{index}.")],
        )
        for index in range(28)
    ]
    gold = [
        {
            "fold_id": index % 10,
            "label": index % 2,
            "statement_id": statement.uuid,
        }
        for index, statement in enumerate(statements)
    ]
    graph = adapter.nx.DiGraph()
    graph.add_nodes_from(statement.get_hash() for statement in statements)
    for target in range(13):
        contributor = 14 + target
        gold[contributor]["fold_id"] = (gold[target]["fold_id"] + 1) % 10
        graph.add_edge(statements[target].get_hash(), statements[contributor].get_hash())
    gold[27]["fold_id"] = gold[13]["fold_id"]
    graph.add_edge(statements[13].get_hash(), statements[27].get_hash())
    audit = adapter._graph_fold_dependency_audit(statements, gold, graph)
    assert audit["cross_fold_descendant_pairs"] == 13
    assert audit["cross_fold_targets_affected"] == 13
    assert audit["same_fold_descendant_pairs"] == 1
    assert audit["same_fold_targets_affected"] == 1
    assert audit["targets_affected"] == 14
    assert audit["fold_isolated_feature_construction"] is False
    assert audit["inductive_evaluation"] is False
    assert audit["label_isolated_model_fitting"] is True
    assert audit["design_label"] == adapter.EVALUATION_DESIGN_LABEL


def test_exact_frozen_reader_identity_stream_matches_existing_comparators() -> None:
    comparison = ROOT / "data/results/indra_paper_comparison_gold_20260717"
    gold_path = comparison / "paper_reader_eligible_released_gold.jsonl"
    gold = [json.loads(line) for line in gold_path.read_text().splitlines()]
    assert len(gold) == adapter.EXPECTED_READER_ROWS
    observed = adapter._ordered_id_sha(gold)
    manifest = json.loads((comparison / "paper_comparison_gold_manifest.json").read_text())
    assert observed == manifest["outputs"]["paper_reader_eligible_released_gold"][
        "ordered_statement_id_sha256"
    ]
    for prediction_path in (
        ROOT
        / "data/results/indra_paper_reproduction_20260717/orig_belief_readers_predictions.jsonl",
        ROOT
        / "data/results/current_indra_bayesian_paper_20260717/current_bayesian_source_oof_readers_only_predictions.jsonl",
        ROOT
        / "data/results/current_indra_hierarchy_paper_20260717/current_simple_hierarchy_readers_only_predictions.jsonl",
    ):
        rows = [json.loads(line) for line in prediction_path.read_text().splitlines()]
        assert len(rows) == adapter.EXPECTED_READER_ROWS
        assert adapter._ordered_id_sha(rows) == observed


def test_bundle_publication_is_transactional_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    rows = {"rows.jsonl": [{"value": 1}]}
    manifest = {"schema_version": 1}
    adapter._publish_bundle(output, rows, manifest)
    assert json.loads((output / "rows.jsonl").read_text()) == {"value": 1}
    assert (output / adapter.MANIFEST_FILENAME).is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        adapter._publish_bundle(output, rows, manifest)
