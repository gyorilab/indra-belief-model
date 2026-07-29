"""Lock the representative first-write unique-pair snapshot."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from export_representative_curations import (  # noqa: E402
    ALLOWED_REPRESENTATIVE_CURATION_SOURCES,
    BENCHMARK_TARGET_UNIQUE_PAIRS,
    DEFAULT_AFTER_ID,
    DEFAULT_CURATOR,
    DEFAULT_DATASET_ID,
    DEFAULT_MANIFEST,
    DEFAULT_META,
    DEFAULT_POOL,
    DEFAULT_POOL_PROVENANCE,
    DEFAULT_PRE_RESERVOIR_MANIFEST,
    DEFAULT_SNAPSHOT,
    PINNED_PRIOR_BENCHMARK_PAIR_SOURCES,
    eligible_curations,
    exact_pair,
    load_pool_provenance,
    resolve_prior_pair_exclusions,
    select_all_earliest_pairs,
    validate_curation_tag,
    validate_export_identity,
)
from indra_belief.curation import aggregate_gold  # noqa: E402
from reservoir_sample_cogex import DEFAULT_SAMPLE_N, SOURCE_POPULATION_ROWS  # noqa: E402

SNAPSHOT = ROOT / "data/benchmark/representative_indra_curations_400.jsonl"
META = ROOT / "data/benchmark/representative_indra_curations_400.meta.json"
MANIFEST = ROOT / "data/benchmark/cogex_representative_pool_manifest.jsonl"
PRE_RESERVOIR_MANIFEST = ROOT / "data/benchmark/mock7ee_pre_reservoir_pair_manifest.jsonl"
EXCLUDED_POOL_PAIRS = {
    (21016737215561966, -1409443675420064898),
    (-23763221908346723, -3811282799351081683),
}
EXPECTED_SNAPSHOT_ROW_KEYS = {
    "matches_hash",
    "source_hash",
    "source_api",
    "pmid",
    "evidence_text",
    "stmt_type",
    "statement",
    "tag",
    "gold",
    "gold_status",
    "curator",
    "curation_source",
    "curation_id",
    "curation_date",
}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def _pair(row: dict) -> tuple[int, int] | None:
    mh = row.get("matches_hash", row.get("pa_hash"))
    sh = row.get("source_hash")
    if mh is None or sh is None:
        return None
    return int(mh), int(sh)


def test_representative_curation_snapshot_is_self_consistent():
    meta = json.loads(META.read_text())
    rows = _rows(SNAPSHOT)
    pairs = [_pair(row) for row in rows]
    excluded_ids = {
        19993,
        19998,
        20135,
        20148,
        20158,
        20169,
        20243,
        20245,
        20279,
        20305,
        20316,
        20320,
    }

    assert meta["schema_version"] == 2
    assert meta["dataset_id"] == DEFAULT_DATASET_ID
    assert meta["benchmark_status"] == "pending"
    assert meta["pair_target_status"] == "complete"
    assert meta["selection"]["scope"] == "all qualifying events returned at export time"
    assert "target_unique_pairs" not in meta["selection"]
    assert meta["selection"]["canonicalization"] == (
        "earliest qualifying submission per exact pair"
    )
    assert meta["selection"]["repeat_handling"] == (
        "exclude later submissions for an already-retained exact pair"
    )
    assert set(meta["selection"]["allowed_curation_sources"]) == set(
        ALLOWED_REPRESENTATIVE_CURATION_SOURCES
    )
    assert meta["selection"]["first_curation_id"] == 19920
    assert meta["selection"]["last_curation_id"] == 20334
    assert meta["selection"]["last_observed_event_id"] == 20334
    assert meta["counts"]["observed_submission_events"] == 415
    assert meta["counts"]["retained_submission_events"] == 403
    assert meta["counts"]["excluded_repeat_submission_events"] == 12
    assert len(rows) == meta["counts"]["unique_pairs"] == 403
    assert len(set(pairs)) == len(rows)
    assert {row["curation_id"] for row in rows} == set(range(19920, 20335)) - excluded_ids
    excluded_events = meta["deduplication_audit"]["excluded_repeat_events"]
    assert [event["curation_id"] for event in excluded_events] == sorted(excluded_ids)
    retained_by_id = {row["curation_id"]: row for row in rows}
    for event in excluded_events:
        retained = retained_by_id[event["retained_curation_id"]]
        assert (event["matches_hash"], event["source_hash"]) == _pair(retained)
        assert event["curation_source"] in ALLOWED_REPRESENTATIVE_CURATION_SOURCES
    assert meta["deduplication_audit"]["repeat_pair_groups"] == 11
    assert meta["deduplication_audit"]["historical_tag_conflict_pairs"] == 3
    assert meta["deduplication_audit"]["historical_binary_conflict_pairs"] == 1

    for row in rows:
        assert set(row) == EXPECTED_SNAPSHOT_ROW_KEYS
        assert not any(isinstance(value, list) for value in row.values())
        assert row["gold"] == aggregate_gold([row["tag"]])
        assert row["gold_status"] == "canonical_first_submission"
        assert row["curator"] == DEFAULT_CURATOR
        assert row["curation_source"] in ALLOWED_REPRESENTATIVE_CURATION_SOURCES
        assert isinstance(row["curation_id"], int)
        assert row["stmt_type"] == row["statement"]["type"]
        assert row["statement"]["matches_hash"] == row["matches_hash"]
        assert isinstance(row["statement"]["matches_hash"], int)
        assert not {"belief", "evidence", "id", "supported_by", "supports"} & row["statement"].keys()
    labels = Counter(row["gold"] for row in rows)
    assert labels == {"correct": 199, "incorrect": 204}
    assert meta["retained_curation_source_counts"] == {
        "indra-belief viewer": 391,
        "indra-belief viewer/representative": 12,
    }
    formerly_conflicted = [row for row in rows if row["curation_id"] == 19928]
    assert len(formerly_conflicted) == 1
    assert formerly_conflicted[0]["tag"] == "wrong_relation"
    assert not any(row["curation_id"] == 19993 for row in rows)

    digest = hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest()
    assert digest == meta["artifact"]["sha256"]


def test_representative_snapshot_is_membership_auditable_from_clean_checkout():
    meta = json.loads(META.read_text())
    manifest = _rows(MANIFEST)
    manifest_pairs = {_pair(row) for row in manifest}
    snapshot_pairs = {_pair(row) for row in _rows(SNAPSHOT)}

    assert len(manifest) == len(manifest_pairs) == meta["representative_pool"]["pairs"] == 5000
    assert snapshot_pairs <= manifest_pairs
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == meta["representative_pool"]["manifest_sha256"]

    prior_pairs: set[tuple[int, int] | None] = set()
    for path in (ROOT / "data/benchmark").glob("*.jsonl"):
        if path in {SNAPSHOT, MANIFEST}:
            continue
        prior_pairs.update(_pair(row) for row in _rows(path))
    prior_pairs.discard(None)
    selected_overlap = snapshot_pairs & prior_pairs
    pool_overlap = manifest_pairs & prior_pairs
    excluded_rows = [row for row in manifest if row.get("excluded_from_curation") is True]
    excluded_pairs = {_pair(row) for row in excluded_rows}
    assert not selected_overlap
    assert snapshot_pairs.isdisjoint(excluded_pairs)
    assert pool_overlap == excluded_pairs == EXCLUDED_POOL_PAIRS
    assert all(row["exclusion_reason"] == "preexisting benchmark exact-pair overlap" for row in excluded_rows)
    assert all(row["exclusion_source_files"] for row in excluded_rows)
    assert meta["contamination"]["selected_pairs_overlapping_preexisting_benchmarks"] == 0
    assert meta["contamination"]["pool_pairs_overlapping_preexisting_benchmarks"] == 2


def test_representative_pool_provenance_names_the_full_source_population():
    meta = json.loads(META.read_text())
    pool = meta["representative_pool"]
    materialization = meta["statement_materialization"]

    assert meta["benchmark_status"] == "pending"
    assert meta["benchmark_target"] == {
        "unit": "unique exact (statement, evidence) pairs",
        "unique_pairs": 400,
    }
    assert meta["counts"]["unique_pairs_remaining_to_benchmark_target"] == 0
    assert meta["counts"]["unique_pairs_above_benchmark_target"] == 3
    assert meta["benchmark_blockers"] == {
        "selection_randomness_unproven": True,
    }
    assert meta["selection_auditability"] == {
        "reservoir_membership_proven": True,
        "historical_draw_log_available": False,
        "historical_sampler_allowed_replacement": True,
        "simple_random_completed_subset_proven": False,
        "reason": (
            "the historical viewer retained no draw/skip log and retried "
            "unmaterializable or textless rows"
        ),
    }
    assert DEFAULT_SAMPLE_N == DEFAULT_POOL_PROVENANCE["sample_size"] == pool["pairs"] == 5_000
    assert SOURCE_POPULATION_ROWS == pool["population_rows"] == 44_944_056
    assert pool["sampling_unit"] == "CoGEx evidence row"
    assert pool["without_replacement"] is True
    assert pool["materialization_sha256"] == DEFAULT_POOL_PROVENANCE["materialization_sha256"]
    assert pool["source_dump_sha256"] == DEFAULT_POOL_PROVENANCE["source_dump_sha256"]
    assert materialization == {
        "endpoint": "/statements/from_hashes",
        "ev_limit": 1,
        "format": "json-js",
        "removed_fields": ["belief", "evidence", "id", "supported_by", "supports"],
        "unique_statements": 403,
    }


def test_pre_reservoir_history_is_frozen_and_disjoint():
    meta = json.loads(META.read_text())
    history = meta["pre_reservoir_curator_history"]
    rows = _rows(PRE_RESERVOIR_MANIFEST)
    history_pairs = {_pair(row) for row in rows}
    snapshot_pairs = {_pair(row) for row in _rows(SNAPSHOT)}

    assert history == {
        "through_curation_id": 19918,
        "source_filter": "indra-belief viewer",
        "raw_submissions": 124,
        "unique_pairs": 123,
        "selected_pair_overlap": 0,
        "manifest_path": "data/benchmark/mock7ee_pre_reservoir_pair_manifest.jsonl",
        "manifest_sha256": hashlib.sha256(PRE_RESERVOIR_MANIFEST.read_bytes()).hexdigest(),
    }
    assert len(rows) == len(history_pairs) == 123
    assert sum(len(row["curation_ids"]) for row in rows) == 124
    assert {cid for row in rows for cid in row["curation_ids"]} == set(range(19795, 19919))
    assert snapshot_pairs.isdisjoint(history_pairs)


def test_excluded_pool_pair_cannot_become_an_eligible_submission():
    allowed_pair = (11, 22)
    off_lane_pair = (33, 44)
    excluded_pair = next(iter(EXCLUDED_POOL_PAIRS))
    rows = [
        {
            "id": 101,
            "curator": "mock7ee@gmail.com",
            "source": "indra-belief viewer/representative",
            "pa_hash": excluded_pair[0],
            "source_hash": excluded_pair[1],
        },
        {
            "id": 102,
            "curator": "mock7ee@gmail.com",
            "source": "indra-belief viewer/representative",
            "pa_hash": allowed_pair[0],
            "source_hash": allowed_pair[1],
        },
        {
            "id": 103,
            "curator": "mock7ee@gmail.com",
            "source": "direct API probe",
            "pa_hash": off_lane_pair[0],
            "source_hash": off_lane_pair[1],
        },
    ]
    eligible = eligible_curations(
        rows,
        curator="mock7ee@gmail.com",
        after_id=100,
        pool_pairs={allowed_pair, excluded_pair, off_lane_pair},
        excluded_pairs={excluded_pair},
    )
    assert [row["id"] for row in eligible] == [102]


def test_default_pool_exclusions_are_pinned_and_new_overlap_fails_closed():
    pinned = set(PINNED_PRIOR_BENCHMARK_PAIR_SOURCES)
    discovered = {
        pair: ["some_current_source.jsonl"]
        for pair in pinned
    }
    assert resolve_prior_pair_exclusions(
        pool_path=DEFAULT_POOL,
        pool_pairs=pinned,
        discovered_sources=discovered,
    ) == PINNED_PRIOR_BENCHMARK_PAIR_SOURCES

    new_overlap = (11, 22)
    with pytest.raises(ValueError, match="new prior-benchmark overlap"):
        resolve_prior_pair_exclusions(
            pool_path=DEFAULT_POOL,
            pool_pairs=pinned | {new_overlap},
            discovered_sources={**discovered, new_overlap: ["later_benchmark.jsonl"]},
        )


def test_noncanonical_selection_cannot_overwrite_canonical_outputs(tmp_path: Path):
    canonical = {
        "email": DEFAULT_CURATOR,
        "after_id": DEFAULT_AFTER_ID,
        "benchmark_target": BENCHMARK_TARGET_UNIQUE_PAIRS,
        "pool": DEFAULT_POOL,
        "pool_provenance": None,
        "dataset_id": DEFAULT_DATASET_ID,
        "out": DEFAULT_SNAPSHOT,
        "meta": DEFAULT_META,
        "manifest": DEFAULT_MANIFEST,
        "pre_reservoir_manifest": DEFAULT_PRE_RESERVOIR_MANIFEST,
    }
    validate_export_identity(**canonical)

    with pytest.raises(ValueError, match="cannot write canonical"):
        validate_export_identity(**{**canonical, "benchmark_target": 50})
    with pytest.raises(ValueError, match="cannot write canonical"):
        validate_export_identity(
            **{**canonical, "pool_provenance": tmp_path / "override.json"}
        )

    custom = {
        **canonical,
        "benchmark_target": 50,
        "dataset_id": "custom_representative_50",
        "out": tmp_path / "snapshot.jsonl",
        "meta": tmp_path / "snapshot.meta.json",
        "manifest": tmp_path / "pool.jsonl",
        "pre_reservoir_manifest": tmp_path / "history.jsonl",
    }
    validate_export_identity(**custom)

    with pytest.raises(ValueError, match="custom --dataset-id"):
        validate_export_identity(**{**custom, "dataset_id": DEFAULT_DATASET_ID})


@pytest.mark.parametrize("target", [0, -1, True, 1.5, "400"])
def test_export_identity_rejects_invalid_benchmark_targets(target):
    with pytest.raises(ValueError, match="benchmark_target must be a positive integer"):
        validate_export_identity(
            email=DEFAULT_CURATOR,
            after_id=DEFAULT_AFTER_ID,
            benchmark_target=target,
            pool=DEFAULT_POOL,
            pool_provenance=None,
            dataset_id=DEFAULT_DATASET_ID,
            out=DEFAULT_SNAPSHOT,
            meta=DEFAULT_META,
            manifest=DEFAULT_MANIFEST,
            pre_reservoir_manifest=DEFAULT_PRE_RESERVOIR_MANIFEST,
        )


@pytest.mark.parametrize("tag", [None, "", "not_a_curation_tag", 42, True])
def test_curation_tag_validation_fails_closed(tag):
    with pytest.raises(ValueError, match="invalid tag"):
        validate_curation_tag({"id": 1, "tag": tag})


def test_curation_tag_validation_accepts_the_canonical_vocabulary():
    assert validate_curation_tag({"id": 1, "tag": "correct"}) == "correct"
    assert validate_curation_tag({"id": 2, "tag": "wrong_relation"}) == "wrong_relation"


def test_all_unique_pair_selection_is_earliest_first_and_scans_the_tail():
    rows = [
        {"id": 7, "pa_hash": 20, "source_hash": 200, "tag": "grounding"},
        {"id": 6, "pa_hash": 40, "source_hash": 400, "tag": "correct"},
        {"id": 2, "pa_hash": "10", "source_hash": "100", "tag": "correct"},
        {"id": 5, "pa_hash": 30, "source_hash": 300, "tag": "grounding"},
        {"id": 1, "pa_hash": 10, "source_hash": 100, "tag": "wrong_relation"},
        {"id": 3, "pa_hash": 20, "source_hash": 200, "tag": "correct"},
    ]
    original = [dict(row) for row in rows]

    retained, repeats, observed, by_pair = select_all_earliest_pairs(rows)

    assert [row["id"] for row in retained] == [1, 3, 5, 6]
    assert [row["id"] for row in repeats] == [2, 7]
    assert [row["id"] for row in observed] == [1, 2, 3, 5, 6, 7]
    assert [row["tag"] for row in by_pair[(10, 100)]] == ["wrong_relation", "correct"]
    assert retained[0]["tag"] == "wrong_relation"
    assert rows == original


def test_unique_pair_selection_returns_an_incomplete_snapshot_without_error():
    rows = [
        {"id": 1, "pa_hash": 10, "source_hash": 100},
        {"id": 2, "pa_hash": 10, "source_hash": 100},
        {"id": 3, "pa_hash": 20, "source_hash": 200},
    ]
    retained, repeats, observed, _ = select_all_earliest_pairs(rows)
    assert [row["id"] for row in retained] == [1, 3]
    assert [row["id"] for row in repeats] == [2]
    assert [row["id"] for row in observed] == [1, 2, 3]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"id": 1.0, "pa_hash": 10, "source_hash": 100}], "exact integer id"),
        ([{"id": 1, "pa_hash": 10, "source_hash": 1.5}], "exact pair hashes"),
        (
            [
                {"id": 1, "pa_hash": 10, "source_hash": 100},
                {"id": "1", "pa_hash": 20, "source_hash": 200},
            ],
            "duplicate curation id 1",
        ),
    ],
)
def test_unique_pair_selection_fails_closed_on_ambiguous_events(rows, message):
    with pytest.raises(ValueError, match=message):
        select_all_earliest_pairs(rows)


@pytest.mark.parametrize(
    "value",
    [True, False, 1.0, -3.5, " 1", "1 ", "+1", "1.0", "", None, [], {}],
)
def test_exact_pair_rejects_noncanonical_hash_values(value):
    assert exact_pair({"matches_hash": value, "source_hash": 2}, "matches_hash") is None
    assert exact_pair({"matches_hash": 1, "source_hash": value}, "matches_hash") is None


@pytest.mark.parametrize(
    ("matches_hash", "source_hash", "expected"),
    [(1, -2, (1, -2)), ("1", "-2", (1, -2)), ("0", "0", (0, 0))],
)
def test_exact_pair_accepts_only_integer_or_signed_digit_string(matches_hash, source_hash, expected):
    assert exact_pair(
        {"matches_hash": matches_hash, "source_hash": source_hash},
        "matches_hash",
    ) == expected


def test_nondefault_pool_requires_and_verifies_explicit_provenance(tmp_path: Path):
    pool = tmp_path / "other-pool.jsonl"
    pool.write_text('{"stmt_hash": 1, "source_hash": 2, "source_api": "test"}\n')
    rows = _rows(pool)

    with pytest.raises(ValueError, match="nondefault --pool requires --pool-provenance"):
        load_pool_provenance(pool, None, rows)

    provenance = dict(DEFAULT_POOL_PROVENANCE)
    provenance.update({
        "sample_size": 1,
        "population_rows": 10,
        "source_dump": "test fixture",
        "source_dump_sha256": "0" * 64,
        "materialization_sha256": hashlib.sha256(pool.read_bytes()).hexdigest(),
    })
    sidecar = tmp_path / "other-pool.meta.json"
    sidecar.write_text(json.dumps(provenance))
    assert load_pool_provenance(pool, sidecar, rows)["population_rows"] == 10

    provenance["materialization_sha256"] = "f" * 64
    sidecar.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="pool SHA-256 mismatch"):
        load_pool_provenance(pool, sidecar, rows)
