from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
from indra.statements import Activation, Agent, Evidence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import score_current_indra_bayesian_paper as adapter  # noqa: E402


def _twenty_statement_reader_panel():
    statements = []
    gold = []
    observations = []
    for index in range(20):
        reach = Evidence(
            source_api="reach",
            text=f"A{index} activates B{index}.",
            annotations={"found_by": "Positive_regulation_syntax_1_verb"},
        )
        signor = Evidence(
            source_api="signor",
            text=f"SIGNOR: A{index} activates B{index}.",
        )
        statement = Activation(
            Agent(f"A{index}"), Agent(f"B{index}"), evidence=[reach, signor]
        )
        statements.append(statement)
        # Each fold has exactly one negative and one positive pair.
        pair_label = index // 10
        _, subtype = adapter._subtype_identity(reach)
        assert subtype is not None
        observations.extend(
            [
                adapter.PairObservation(
                    adjudication_id=f"reach-{index}",
                    eligible_position=index,
                    label=pair_label,
                    source="reach",
                    source_hash=str(reach.get_source_hash()),
                    concrete_subtypes=(subtype,),
                    raw_entry_count=1,
                ),
                adapter.PairObservation(
                    adjudication_id=f"signor-{index}",
                    eligible_position=index,
                    label=1 - pair_label,
                    source="signor",
                    source_hash=str(signor.get_source_hash()),
                    concrete_subtypes=(),
                    raw_entry_count=1,
                ),
            ]
        )
        gold.append(
            {
                "fold_id": index % 10,
                "label": index % 2,
                "statement_id": statement.uuid,
            }
        )
    panel = adapter.Panel(
        panel_id=adapter.PANEL_READERS,
        sources=adapter.READER_SOURCES,
        positions=tuple(range(20)),
        gold_rows=tuple(gold),
    )
    return statements, panel, observations


def test_bayesian_reader_crossfit_excludes_test_pairs_and_database_sources() -> None:
    statements, panel, observations = _twenty_statement_reader_panel()
    predictions, provenance, fits = adapter._bayesian_arm(
        adapter.ARM_BAYES_SOURCE_READERS,
        panel,
        statements,
        observations,
        use_subtypes=False,
    )

    assert len(predictions) == len(provenance) == 20
    assert len(fits) == 10
    assert all(set(row) == {"statement_id", "probability_correct"} for row in predictions)
    # One Reach observation from each class is excluded in every test fold.
    assert all(row["train_reviewed_pairs"] == 18 for row in fits)
    assert all(row["excluded_test_reviewed_pairs"] == 2 for row in fits)
    assert all(row["train_source_counts"] == {"reach": [9, 9]} for row in fits)
    assert all(row["train_subtype_counts"] == {} for row in fits)
    assert all(row["no_pseudocounts"] is True for row in fits)
    assert all(
        row["bundled_default_fallback_sources"]
        == sorted(set(adapter.READER_SOURCES) - {"reach"})
        for row in fits
    )
    assert all(row["projected_source_counts"] == {"reach": 1} for row in provenance)
    assert all(row["projected_raw_evidence_count"] == 1 for row in provenance)
    assert all(row["canonical_raw_evidence_count"] == 2 for row in provenance)
    # p/(p+n)=0.5 -> current random error .45; one positive evidence -> belief .5.
    assert [row["probability_correct"] for row in predictions] == pytest.approx(
        [0.5] * 20, abs=1e-15
    )


def test_bayesian_predictions_do_not_consume_statement_gold_labels() -> None:
    statements, panel, observations = _twenty_statement_reader_panel()
    original, _, original_fits = adapter._bayesian_arm(
        adapter.ARM_BAYES_SUBTYPE_READERS,
        panel,
        statements,
        observations,
        use_subtypes=True,
    )
    flipped_gold = tuple(
        {**row, "label": 1 - row["label"]} for row in panel.gold_rows
    )
    flipped_panel = copy.copy(panel)
    object.__setattr__(flipped_panel, "gold_rows", flipped_gold)
    flipped, _, flipped_fits = adapter._bayesian_arm(
        adapter.ARM_BAYES_SUBTYPE_READERS,
        flipped_panel,
        statements,
        observations,
        use_subtypes=True,
    )

    assert original == flipped
    assert original_fits == flipped_fits


def test_pair_subtype_identity_counts_pair_once_per_concrete_variant() -> None:
    first = Evidence(
        source_api="reach",
        text="A activates B.",
        annotations={"found_by": "binding11"},
    )
    second = Evidence(
        source_api="reach",
        text="A activates B.",
        annotations={"found_by": "binding3"},
    )
    assert first.get_source_hash() == second.get_source_hash()
    statement = Activation(Agent("A"), Agent("B"), evidence=[first, second])
    row = {
        "adjudication_id": "s0000-e00000",
        "canonical_corpus_row_index": 7,
        "conflict_resolution": None,
        "corpus_evidence_entry_count": 2,
        "corpus_evidence_json_sha256s": ["a" * 64, "b" * 64],
        "corpus_evidence_positions": [0, 1],
        "corpus_evidence_text_present": [True, True],
        "curation_count": 1,
        "curations": [{"tag_label": 1}],
        "eligible_position": 0,
        "evidence_gold_label": 1,
        "identity_kind": "statement_source_hash_pair",
        "needed_to_resolve_statement": False,
        "paper_statement_hash": str(statement.get_hash(shallow=True)),
        "queue_item_id": None,
        "review_status": "positive",
        "same_pair_conflict": False,
        "source_apis": ["reach"],
        "source_hash": str(first.get_source_hash()),
    }
    targets = [{"canonical_corpus_row_index": 7}]

    observations, summary = adapter._pair_observations([row], [statement], targets)
    source_counts, subtype_counts = adapter._count_pairs(observations)

    assert len(observations) == 1
    assert len(observations[0].concrete_subtypes) == 2
    assert source_counts == {"reach": [1, 0]}
    assert sum(counts[0] for counts in subtype_counts["reach"].values()) == 2
    assert summary["multi_concrete_subtype_pairs"] == 1
    assert summary["concrete_subtype_pair_cells"] == 2


def test_simple_reader_primary_and_all_evidence_sensitivity_are_distinct() -> None:
    statements, panel, _ = _twenty_statement_reader_panel()
    reader_predictions, reader_provenance = adapter._simple_arm(
        adapter.ARM_SIMPLE_READERS,
        panel,
        statements,
        adapter.READER_SOURCES,
    )
    sensitivity_predictions, sensitivity_provenance = adapter._simple_arm(
        adapter.ARM_SIMPLE_READER_SENSITIVITY,
        panel,
        statements,
        adapter.ALL_SOURCES,
    )

    assert any(
        left["probability_correct"] != right["probability_correct"]
        for left, right in zip(reader_predictions, sensitivity_predictions, strict=True)
    )
    assert all(row["projected_raw_evidence_count"] == 1 for row in reader_provenance)
    assert all(row["projected_raw_evidence_count"] == 2 for row in sensitivity_provenance)
