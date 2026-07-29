#!/usr/bin/env python3
"""Run current INDRA hierarchy-propagated SimpleScorer on the paper graph.

The verified 2023 release pickle is used because it contains the frozen object
support graph.  After verifying the pickle's release digest, this adapter
unpickles it under INDRA 1.24.0, invokes the current official
``build_refinements_graph`` and ``BeliefEngine.get_hierarchy_probs`` paths, and
emits statement probabilities for the exact paper panels.

The primary all-source arm uses all direct evidence plus non-negated evidence
in graph descendants.  The primary reader arm traverses the identical frozen
graph but restricts direct and inherited evidence to the five frozen readers.
An all-evidence row subset is separately labelled as a reader-eligibility
sensitivity and is not presented as paper reader-input parity.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import inspect
import json
import math
import networkx as nx
import os
import pickle
import platform
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from indra.belief import (
    BeliefEngine,
    SimpleScorer,
    build_refinements_graph,
    get_ev_for_stmts_from_supports,
)
from indra.statements import Evidence, Statement

import score_current_indra_bayesian_paper as bayes
import score_current_indra_simple_paper as base


SCHEMA_VERSION = 1
ARTIFACT_KIND = "current_indra_simple_hierarchy_paper_predictions"
ARM_ALL = "indra_1.24.0_simple_hierarchy_all_sources"
ARM_READERS = "indra_1.24.0_simple_hierarchy_readers_only"
ARM_READER_SENSITIVITY = (
    "indra_1.24.0_simple_hierarchy_reader_eligible_all_evidence_sensitivity"
)
PREDICTION_FILENAMES = {
    ARM_ALL: "current_simple_hierarchy_all_sources_predictions.jsonl",
    ARM_READERS: "current_simple_hierarchy_readers_only_predictions.jsonl",
    ARM_READER_SENSITIVITY: (
        "current_simple_hierarchy_reader_eligible_all_evidence_sensitivity_predictions.jsonl"
    ),
}
PROVENANCE_FILENAME = "current_simple_hierarchy_prediction_provenance.jsonl"
FOLD_METRICS_FILENAME = "current_simple_hierarchy_diagnostic_fold_metrics.jsonl"
MANIFEST_FILENAME = "current_simple_hierarchy_paper_manifest.json"
EXPECTED_PICKLE_ROOTS = 894_939
EXPECTED_GRAPH_NODES = 895_459
EXPECTED_GRAPH_EDGES = 637_573


class ContractError(ValueError):
    """Raised when release identity, graph, projection, or coverage fails."""


def _pickle_descriptor(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ContractError("paper manifest files must be an array")
    matches = [
        row
        for row in files
        if isinstance(row, dict) and row.get("filename") == "indra_benchmark_corpus.pkl"
    ]
    if len(matches) != 1:
        raise ContractError("paper manifest must identify one canonical pickle")
    descriptor = matches[0]
    observed_sha = base._sha256(path)
    if descriptor.get("sha256") != observed_sha or descriptor.get("bytes") != path.stat().st_size:
        raise ContractError("paper pickle size/digest differs from release manifest")
    if descriptor.get("canonical_for_historical_object_and_refinement_parity") is not True:
        raise ContractError("paper manifest does not designate pickle for graph parity")
    return {
        "path": base._display_path(path),
        "bytes": path.stat().st_size,
        "sha256": observed_sha,
        "verification": "pass_before_trusted_unpickle",
        "unpickle_security_scope": "digest-verified official Zenodo release object",
    }


def _load_verified_pickle(path: Path) -> list[Statement]:
    try:
        with path.open("rb") as stream:
            value = pickle.load(stream)
    except (OSError, pickle.UnpicklingError, EOFError) as exc:
        raise ContractError(f"could not load verified paper pickle: {exc}") from exc
    if not isinstance(value, list) or len(value) != EXPECTED_PICKLE_ROOTS:
        raise ContractError(
            f"pickle root must be a {EXPECTED_PICKLE_ROOTS:,}-statement list"
        )
    if any(not isinstance(statement, Statement) for statement in value):
        raise ContractError("pickle root contains a non-Statement object")
    return value


def _select_targets(
    roots: Sequence[Statement], targets: Sequence[dict[str, Any]]
) -> list[Statement]:
    target_ids = {row["statement_id"] for row in targets}
    if len(target_ids) != len(targets):
        raise ContractError("target statement UUIDs are not unique")
    selected: dict[str, Statement] = {}
    for statement in roots:
        if statement.uuid not in target_ids:
            continue
        if statement.uuid in selected:
            raise ContractError(f"pickle repeats target UUID {statement.uuid}")
        selected[statement.uuid] = statement
    missing = sorted(target_ids - set(selected))
    if missing:
        raise ContractError(f"pickle is missing target UUIDs: {missing[:3]}")
    ordered = [selected[row["statement_id"]] for row in targets]
    for row, statement in zip(targets, ordered, strict=True):
        if str(statement.get_hash(shallow=True)) != row["matches_hash"]:
            raise ContractError(f"{row['statement_id']}: pickle/target shallow hash mismatch")
    return ordered


def _project_statement(statement: Statement, sources: set[str]) -> Statement:
    projected = copy.copy(statement)
    projected.evidence = [ev for ev in statement.evidence if ev.source_api in sources]
    return projected


def _project_extra(evidence: Sequence[Evidence], sources: set[str]) -> list[Evidence]:
    return [item for item in evidence if item.source_api in sources]


def _prediction(statement_id: str, value: float) -> dict[str, Any]:
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ContractError(f"{statement_id}: invalid hierarchy probability {probability!r}")
    return {"probability_correct": probability, "statement_id": statement_id}


def _score_reader_projection(
    statements: Sequence[Statement], extra_evidence: Sequence[Sequence[Evidence]]
) -> list[float]:
    allowed = set(bayes.READER_SOURCES)
    projected_statements = [_project_statement(statement, allowed) for statement in statements]
    if any(not statement.evidence for statement in projected_statements):
        raise ContractError("reader projection yielded an evidence-free eligible statement")
    projected_extra = [_project_extra(items, allowed) for items in extra_evidence]
    scorer = SimpleScorer()
    scorer.check_prior_probs(projected_statements)
    return [
        float(value)
        for value in scorer.score_statements(projected_statements, projected_extra)
    ]


def _write_bundle(
    output_dir: Path,
    rows_by_name: dict[str, list[dict[str, Any]]],
    manifest: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    final = {name: output_dir / name for name in rows_by_name}
    final[MANIFEST_FILENAME] = output_dir / MANIFEST_FILENAME
    existing = [str(path) for path in final.values() if os.path.lexists(path)]
    if existing:
        raise FileExistsError("refusing to overwrite existing outputs: " + ", ".join(existing))
    with tempfile.TemporaryDirectory(prefix=".current-hierarchy-", dir=output_dir) as tmp:
        stage = Path(tmp)
        staged: dict[str, Path] = {}
        for name, rows in rows_by_name.items():
            path = stage / name
            base._write_jsonl(path, rows)
            staged[name] = path
        manifest_path = stage / MANIFEST_FILENAME
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        staged[MANIFEST_FILENAME] = manifest_path
        published: list[Path] = []
        try:
            for name in [*rows_by_name, MANIFEST_FILENAME]:
                os.link(staged[name], final[name])
                published.append(final[name])
        except BaseException:
            for path in reversed(published):
                path.unlink(missing_ok=True)
            raise


def materialize(
    *,
    pickle_path: Path,
    paper_manifest_path: Path,
    targets_path: Path,
    comparison_manifest_path: Path,
    all_gold_path: Path,
    reader_gold_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    (
        pickle_path,
        paper_manifest_path,
        targets_path,
        comparison_manifest_path,
        all_gold_path,
        reader_gold_path,
    ) = [
        path.resolve()
        for path in (
            pickle_path,
            paper_manifest_path,
            targets_path,
            comparison_manifest_path,
            all_gold_path,
            reader_gold_path,
        )
    ]
    output_dir = output_dir.resolve()
    paper_manifest = base._read_json(paper_manifest_path)
    comparison_manifest = base._read_json(comparison_manifest_path)
    targets = base._read_targets(targets_path)
    all_gold = bayes._read_gold(all_gold_path)
    reader_gold = bayes._read_gold(reader_gold_path)
    pickle_input = _pickle_descriptor(pickle_path, paper_manifest)
    target_input = bayes._verify_output_descriptor(
        targets_path, comparison_manifest, "paper_prediction_targets", targets
    )
    all_gold_input = bayes._verify_output_descriptor(
        all_gold_path, comparison_manifest, "paper_released_gold", all_gold
    )
    reader_gold_input = bayes._verify_output_descriptor(
        reader_gold_path,
        comparison_manifest,
        "paper_reader_eligible_released_gold",
        reader_gold,
    )
    if len(targets) != 1689 or len(all_gold) != 1689 or len(reader_gold) != 1676:
        raise ContractError("frozen panel sizes must be exactly 1689 and 1676")
    if [row["statement_id"] for row in all_gold] != [row["statement_id"] for row in targets]:
        raise ContractError("all-source gold/target order mismatch")
    reader_positions = [index for index, row in enumerate(targets) if row["reader_eligible"]]
    if [row["statement_id"] for row in reader_gold] != [
        targets[index]["statement_id"] for index in reader_positions
    ]:
        raise ContractError("reader gold/target order mismatch")

    roots = _load_verified_pickle(pickle_path)
    statements = _select_targets(roots, targets)
    if sum(len(statement.evidence) for statement in statements) != 34_035:
        raise ContractError("pickle target evidence count differs from frozen corpus")

    graph = build_refinements_graph(roots)
    graph_nodes = graph.number_of_nodes()
    graph_edges = graph.number_of_edges()
    if (graph_nodes, graph_edges) != (EXPECTED_GRAPH_NODES, EXPECTED_GRAPH_EDGES):
        raise ContractError(
            f"current graph shape {(graph_nodes, graph_edges)} differs from frozen expectation"
        )
    if not nx.is_directed_acyclic_graph(graph):
        raise ContractError("current graph builder produced a cycle")

    scorer = SimpleScorer()
    engine = BeliefEngine(scorer, refinements_graph=graph)
    all_by_hash = engine.get_hierarchy_probs(statements)
    all_scores = [float(all_by_hash[statement.get_hash()]) for statement in statements]
    if len(all_by_hash) != len(statements):
        raise ContractError("target statement hashes collide in hierarchy output")

    # Obtain official inherited evidence lists separately for provenance and
    # the five-reader projection, then prove they reproduce the engine result.
    extra_evidence = get_ev_for_stmts_from_supports(statements, graph)
    direct_recheck = [
        float(value) for value in scorer.score_statements(statements, extra_evidence)
    ]
    if np.asarray(all_scores, dtype="<f8").tobytes() != np.asarray(
        direct_recheck, dtype="<f8"
    ).tobytes():
        raise ContractError("BeliefEngine and official evidence-helper scores diverge")

    reader_statements = [statements[index] for index in reader_positions]
    reader_extra = [extra_evidence[index] for index in reader_positions]
    reader_scores = _score_reader_projection(reader_statements, reader_extra)
    sensitivity_scores = [all_scores[index] for index in reader_positions]

    predictions = {
        ARM_ALL: [
            _prediction(gold["statement_id"], score)
            for gold, score in zip(all_gold, all_scores, strict=True)
        ],
        ARM_READERS: [
            _prediction(gold["statement_id"], score)
            for gold, score in zip(reader_gold, reader_scores, strict=True)
        ],
        ARM_READER_SENSITIVITY: [
            _prediction(gold["statement_id"], score)
            for gold, score in zip(reader_gold, sensitivity_scores, strict=True)
        ],
    }

    provenance: list[dict[str, Any]] = []
    for panel_arm, positions, scores, sources in (
        (ARM_ALL, list(range(len(statements))), all_scores, None),
        (ARM_READERS, reader_positions, reader_scores, set(bayes.READER_SOURCES)),
        (ARM_READER_SENSITIVITY, reader_positions, sensitivity_scores, None),
    ):
        panel_gold = all_gold if panel_arm == ARM_ALL else reader_gold
        for gold, position, score in zip(panel_gold, positions, scores, strict=True):
            statement = statements[position]
            extras = extra_evidence[position]
            direct_projected = [
                ev for ev in statement.evidence if sources is None or ev.source_api in sources
            ]
            extras_projected = [
                ev for ev in extras if sources is None or ev.source_api in sources
            ]
            combined_unique = set(direct_projected)
            combined_unique.update(extras_projected)
            descendants = nx.descendants(graph, statement.get_hash())
            provenance.append(
                {
                    "arm_id": panel_arm,
                    "descendant_statement_count": len(descendants),
                    "direct_canonical_evidence_count": len(statement.evidence),
                    "direct_projected_evidence_count": len(direct_projected),
                    "inherited_nonnegated_evidence_count": len(extras),
                    "inherited_projected_nonnegated_evidence_count": len(extras_projected),
                    "input_sources": (
                        "all sources present in frozen graph"
                        if sources is None
                        else list(bayes.READER_SOURCES)
                    ),
                    "probability_correct": float(score),
                    "projected_combined_source_counts": dict(
                        sorted(
                            Counter(
                                ev.source_api for ev in [*direct_projected, *extras_projected]
                            ).items()
                        )
                    ),
                    "scorer_visible_unique_evidence_count": len(combined_unique),
                    "statement_id": gold["statement_id"],
                }
            )

    fold_metrics: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for arm_id, rows in predictions.items():
        gold = all_gold if arm_id == ARM_ALL else reader_gold
        arm_folds, summary = bayes._diagnostics(arm_id, rows, gold)
        fold_metrics.extend(arm_folds)
        diagnostics[arm_id] = summary

    rows_by_name = {
        **{PREDICTION_FILENAMES[arm_id]: rows for arm_id, rows in predictions.items()},
        PROVENANCE_FILENAME: provenance,
        FOLD_METRICS_FILENAME: fold_metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".current-hierarchy-desc-", dir=output_dir) as tmp:
        stage = Path(tmp)
        output_descriptors: dict[str, Any] = {}
        for name, rows in rows_by_name.items():
            path = stage / name
            base._write_jsonl(path, rows)
            output_descriptors[name] = {
                "path": base._display_path(output_dir / name),
                "bytes": path.stat().st_size,
                "rows": len(rows),
                "sha256": base._sha256(path),
            }

    module_path = Path(inspect.getsourcefile(SimpleScorer) or "").resolve()
    resource = module_path.parent.parent / "resources" / "default_belief_probs.json"
    direct_simple_path = (
        Path.cwd()
        / "data/results/current_indra_bayesian_paper_20260717/current_simple_direct_all_sources_predictions.jsonl"
    )
    direct_simple = None
    if direct_simple_path.is_file():
        direct_rows = bayes._read_jsonl(direct_simple_path)
        direct_map = {row["statement_id"]: row["probability_correct"] for row in direct_rows}
        changed = sum(
            not math.isclose(
                direct_map[prediction["statement_id"]],
                prediction["probability_correct"],
                rel_tol=0.0,
                abs_tol=0.0,
            )
            for prediction in predictions[ARM_ALL]
        )
        direct_simple = {
            "path": base._display_path(direct_simple_path),
            "sha256": base._sha256(direct_simple_path),
            "statements_changed_by_hierarchy": changed,
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "claim_scope": {
            "lane": "P",
            "prediction_unit": "assembled_statement",
            "output_semantics": "probability_statement_correct",
            "paid_inference_calls": 0,
            "diagnostics_are_formal_three_family_comparison": False,
        },
        "arms": [
            {
                "arm_id": ARM_ALL,
                "class": "indra.belief.SimpleScorer via indra.belief.BeliefEngine",
                "rows": 1689,
                "input_projection": "all direct plus all non-negated descendant evidence in frozen graph",
                "hierarchy_propagation": True,
            },
            {
                "arm_id": ARM_READERS,
                "class": "indra.belief.SimpleScorer with official hierarchy evidence traversal",
                "rows": 1676,
                "input_projection": "direct plus non-negated descendant evidence restricted to five frozen readers",
                "input_sources": list(bayes.READER_SOURCES),
                "hierarchy_propagation": True,
            },
            {
                "arm_id": ARM_READER_SENSITIVITY,
                "class": "indra.belief.SimpleScorer via indra.belief.BeliefEngine",
                "rows": 1676,
                "input_projection": "all-evidence hierarchy scores subset to reader-eligible rows",
                "hierarchy_propagation": True,
                "direct_reader_parity": False,
                "sensitivity_only": True,
            },
        ],
        "graph_execution": {
            "source": "digest-verified official paper pickle object graph",
            "root_statements": len(roots),
            "nodes": graph_nodes,
            "edges": graph_edges,
            "acyclic": True,
            "builder": "indra.belief.build_refinements_graph",
            "all_source_prediction_path": "indra.belief.BeliefEngine.get_hierarchy_probs",
            "reader_prediction_path": (
                "official get_ev_for_stmts_from_supports traversal, frozen five-source filter, "
                "then current SimpleScorer.score_statements with extra_evidence"
            ),
            "direction": "less detailed to more detailed; descendants contribute evidence",
            "inherited_negated_evidence": "excluded by current official helper",
            "ontology_recomputation": False,
            "current_ontology_used": False,
            "engine_helper_byte_equality_check": "pass",
        },
        "coverage": {
            "all_source_statements": 1689,
            "reader_statements": 1676,
            "canonical_direct_evidence_entries": 34_035,
            "statements_with_descendants": sum(
                bool(nx.descendants(graph, statement.get_hash())) for statement in statements
            ),
            "target_descendant_statement_links": sum(
                len(nx.descendants(graph, statement.get_hash())) for statement in statements
            ),
            "missing_predictions": 0,
            "invalid_predictions": 0,
        },
        "inputs": {
            "canonical_object_graph_pickle": pickle_input,
            "paper_manifest": {
                "path": base._display_path(paper_manifest_path),
                "bytes": paper_manifest_path.stat().st_size,
                "sha256": base._sha256(paper_manifest_path),
            },
            "prediction_targets": target_input,
            "comparison_manifest": {
                "path": base._display_path(comparison_manifest_path),
                "bytes": comparison_manifest_path.stat().st_size,
                "sha256": base._sha256(comparison_manifest_path),
            },
            "all_source_gold_and_folds": all_gold_input,
            "reader_gold_and_folds": reader_gold_input,
            "direct_simple_crosscheck": direct_simple,
        },
        "implementation": {
            "indra_version": importlib.metadata.version("indra"),
            "indra_belief_module": {
                "path": base._display_path(module_path),
                "sha256": base._sha256(module_path),
            },
            "bundled_default_prior_resource": {
                "path": base._display_path(resource),
                "sha256": base._sha256(resource),
            },
            "adapter": {
                "path": base._display_path(Path(__file__)),
                "sha256": base._sha256(Path(__file__)),
            },
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "networkx": nx.__version__,
                "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
                "executable": sys.executable,
            },
        },
        "diagnostic_metrics": diagnostics,
        "runtime_observation": {
            "wall_seconds": time.perf_counter() - started,
            "inference_usd": 0.0,
            "cost_scope": "local CPU execution; released graph acquisition and human curation costs excluded",
        },
        "outputs": output_descriptors,
    }
    _write_bundle(output_dir, rows_by_name, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pickle", type=Path, required=True)
    parser.add_argument("--paper-manifest", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--comparison-manifest", type=Path, required=True)
    parser.add_argument("--all-gold", type=Path, required=True)
    parser.add_argument("--reader-gold", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = materialize(
        pickle_path=args.pickle,
        paper_manifest_path=args.paper_manifest,
        targets_path=args.targets,
        comparison_manifest_path=args.comparison_manifest,
        all_gold_path=args.all_gold,
        reader_gold_path=args.reader_gold,
        output_dir=args.output_dir,
    )
    print(json.dumps({"coverage": manifest["coverage"], "diagnostics": manifest["diagnostic_metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
