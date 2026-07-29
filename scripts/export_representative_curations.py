"""Freeze an auditable representative INDRA unique-pair curation snapshot.

The live INDRA list-all endpoint stores curation identity but not the viewer's
selected dataset.  Provenance is therefore established by an exact
``(pa_hash, source_hash)`` join to the 5,000-pair CoGEx reservoir used by
``/curate?dataset=representative``.  The pre-reservoir local snapshot ended at
curation id 19918.  Every later matching submission returned at export time is
scanned in curation-id order and only the first submission for each exact pair
is retained.  The artifact therefore contains the complete current unique-pair
snapshot; 400 is its benchmark target, not an export cap.

Outputs are small tracked evidence artifacts; the 4.4 MB materialization pool
and 4.5 GB source dump remain ignored.  Later submissions for an already-seen
pair are excluded from the canonical rows rather than treated as independent
votes.  The retained row records one tag, id, and date.  Conflicts found in the
excluded repeat events remain visible as aggregate audit metadata but do not
change the first-submission label.  The genuine pre-reservoir viewer history is
frozen as a separate exact-pair manifest so the zero-overlap claim is
reproducible.

The default pool has a frozen, verified provenance contract below.  A different
``--pool`` must supply ``--pool-provenance`` rather than inheriting the default
dump, seed, population size, or hash by accident.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/export_representative_curations.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from indra_belief.curation import CURATION_TAGS, is_gold_correct  # noqa: E402
from pull_my_curations import _load_env, pull_all  # noqa: E402

DEFAULT_POOL = ROOT / "data/corpora/cogex_evidence_sample.jsonl"
DEFAULT_SNAPSHOT = ROOT / "data/benchmark/representative_indra_curations_400.jsonl"
DEFAULT_META = ROOT / "data/benchmark/representative_indra_curations_400.meta.json"
DEFAULT_MANIFEST = ROOT / "data/benchmark/cogex_representative_pool_manifest.jsonl"
DEFAULT_PRE_RESERVOIR_MANIFEST = (
    ROOT / "data/benchmark/mock7ee_pre_reservoir_pair_manifest.jsonl"
)
PRE_RESERVOIR_SOURCE = "indra-belief viewer"
BENCHMARK_TARGET_UNIQUE_PAIRS = 400
DEFAULT_DATASET_ID = "representative_indra_curations_400"
DEFAULT_CURATOR = "mock7ee@gmail.com"
DEFAULT_AFTER_ID = 19918
ALLOWED_REPRESENTATIVE_CURATION_SOURCES = frozenset({
    "indra-belief viewer",
    "indra-belief viewer/representative",
})
PINNED_PRIOR_BENCHMARK_PAIR_SOURCES = {
    (21016737215561966, -1409443675420064898): [
        "belief_benchmark.jsonl",
        "eval_curation_v1.jsonl",
        "probe_relation_logic.jsonl",
    ],
    (-23763221908346723, -3811282799351081683): [
        "belief_benchmark.jsonl",
    ],
}

DEFAULT_POOL_PROVENANCE = {
    "algorithm": "uniform streaming reservoir sample (Algorithm R)",
    "sampling_unit": "CoGEx evidence row",
    "without_replacement": True,
    "seed": 20260701,
    "sample_size": 5_000,
    "population_rows": 44_944_056,
    "source_dump": "CoGEx 2025-09-16 nodes_Evidence.tsv.gz",
    "source_dump_sha256": "29cee1b4a9367c3a9aa7c9e34066fd679381ebde324a770dae1e4944cef33ff5",
    "materialization_sha256": "4382d25fca3a5d07053be225978d2909f4fc7dc4f2e1dea475c29dcae38b1b3d",
}
POOL_PROVENANCE_FIELDS = frozenset(DEFAULT_POOL_PROVENANCE)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SIGNED_INTEGER_RE = re.compile(r"^-?\d+$")
STATEMENT_BATCH_SIZE = 20
STATEMENT_FIELDS_REMOVED = frozenset({"belief", "evidence", "id", "supported_by", "supports"})
# Generated scratch duplicate excluded by .gitignore; cite canonical tracked
# benchmark sources in the clean-checkout manifest instead.
NONCANONICAL_BENCHMARK_FILES = frozenset({"eval_curation_v1_clean.jsonl"})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def path_label(path: Path) -> str:
    """Prefer a repo-relative artifact label, while supporting external inputs."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def exact_integer(value: object) -> int | None:
    """Parse a canonical signed integer without bool/float coercion."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and SIGNED_INTEGER_RE.fullmatch(value):
        return int(value)
    return None


def exact_pair(row: dict, statement_field: str) -> tuple[int, int] | None:
    matches_hash = exact_integer(row.get(statement_field))
    source_hash = exact_integer(row.get("source_hash"))
    if matches_hash is None or source_hash is None:
        return None
    return matches_hash, source_hash


def load_pool(path: Path) -> tuple[list[dict], dict[tuple[int, int], dict]]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    by_pair: dict[tuple[int, int], dict] = {}
    for row in rows:
        key = exact_pair(row, "stmt_hash")
        if key is None:
            raise ValueError("representative pool contains a row without exact hashes")
        if key in by_pair:
            raise ValueError(f"representative pool contains duplicate exact pair {key}")
        by_pair[key] = row
    return rows, by_pair


def load_pool_provenance(
    pool_path: Path,
    provenance_path: Path | None,
    pool_rows: list[dict],
) -> dict:
    """Load and verify provenance for an exact materialized reservoir.

    Algorithm, source identity, and seed cannot be inferred from a JSONL.  The
    known default is pinned in this script; every other pool must make those
    claims explicitly in a JSON sidecar.  Hash and row count are always checked
    against the actual pool, so even explicit provenance cannot silently drift.
    """
    if provenance_path is not None:
        try:
            raw = json.loads(provenance_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read --pool-provenance {provenance_path}: {exc}") from exc
    elif pool_path.resolve() == DEFAULT_POOL.resolve():
        raw = dict(DEFAULT_POOL_PROVENANCE)
    else:
        raise ValueError(
            "a nondefault --pool requires --pool-provenance with explicit "
            f"fields: {', '.join(sorted(POOL_PROVENANCE_FIELDS))}"
        )

    if not isinstance(raw, dict):
        raise ValueError("pool provenance must be a JSON object")
    missing = POOL_PROVENANCE_FIELDS - raw.keys()
    if missing:
        raise ValueError(f"pool provenance missing fields: {', '.join(sorted(missing))}")

    for field in ("algorithm", "sampling_unit", "source_dump"):
        if not isinstance(raw[field], str) or not raw[field].strip():
            raise ValueError(f"pool provenance {field} must be a non-empty string")
    if raw["without_replacement"] is not True:
        raise ValueError("pool provenance must assert without_replacement=true")
    for field in ("seed", "sample_size", "population_rows"):
        if isinstance(raw[field], bool) or not isinstance(raw[field], int) or raw[field] < 1:
            raise ValueError(f"pool provenance {field} must be a positive integer")
    for field in ("source_dump_sha256", "materialization_sha256"):
        if not isinstance(raw[field], str) or not SHA256_RE.fullmatch(raw[field]):
            raise ValueError(f"pool provenance {field} must be a lowercase SHA-256")

    if raw["sample_size"] != len(pool_rows):
        raise ValueError(
            f"pool row-count mismatch: materialization has {len(pool_rows)}, "
            f"provenance claims {raw['sample_size']}"
        )
    if raw["population_rows"] < raw["sample_size"]:
        raise ValueError("pool provenance population_rows cannot be smaller than sample_size")
    actual_sha256 = sha256(pool_path)
    if raw["materialization_sha256"] != actual_sha256:
        raise ValueError(
            "pool SHA-256 mismatch: "
            f"materialization is {actual_sha256}, provenance claims {raw['materialization_sha256']}"
        )

    # Return only the contract fields, with the verified materialization digest.
    return {field: raw[field] for field in DEFAULT_POOL_PROVENANCE}


def freeze_statement_structure(statement: dict, matches_hash: int) -> dict:
    """Keep the extraction semantics while dropping evidence and volatile fields."""
    frozen = {
        key: value
        for key, value in statement.items()
        if key not in STATEMENT_FIELDS_REMOVED
    }
    frozen["matches_hash"] = matches_hash
    if not isinstance(frozen.get("type"), str) or not frozen["type"]:
        raise ValueError(f"statement {matches_hash} has no type")
    return frozen


def fetch_statement_structures(
    base_url: str,
    matches_hashes: set[int],
    *,
    batch_size: int = STATEMENT_BATCH_SIZE,
    retries: int = 4,
) -> dict[int, dict]:
    """Batch-materialize exact statement structures from the public INDRA API."""
    import httpx

    wanted = sorted(matches_hashes)
    found: dict[int, dict] = {}
    with httpx.Client(
        base_url=base_url,
        timeout=httpx.Timeout(120.0, connect=30.0),
        headers={"User-Agent": "indra-belief-representative-freeze/1"},
    ) as client:
        for start in range(0, len(wanted), batch_size):
            chunk = wanted[start:start + batch_size]
            last_error = "no response"
            for attempt in range(retries + 1):
                try:
                    response = client.post(
                        "/statements/from_hashes",
                        json={"hashes": chunk},
                        params={"format": "json-js", "ev_limit": 1},
                    )
                    if response.status_code == 200:
                        payload = response.json()
                        results = payload.get("results", {}) if isinstance(payload, dict) else {}
                        items = results.items() if isinstance(results, dict) else enumerate(results)
                        for result_key, statement in items:
                            if not isinstance(statement, dict):
                                continue
                            raw_hash = statement.get("matches_hash", result_key)
                            try:
                                matches_hash = int(raw_hash)
                            except (TypeError, ValueError):
                                continue
                            if matches_hash in matches_hashes:
                                found[matches_hash] = freeze_statement_structure(statement, matches_hash)
                        last_error = "response omitted requested hashes"
                        break
                    last_error = f"HTTP {response.status_code}"
                except (httpx.HTTPError, json.JSONDecodeError) as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                if attempt < retries:
                    time.sleep(0.5 * (2 ** attempt))
            else:
                raise RuntimeError(f"statement batch starting {chunk[0]} failed: {last_error}")

    missing = matches_hashes - found.keys()
    if missing:
        preview = ", ".join(str(value) for value in sorted(missing)[:5])
        raise RuntimeError(
            f"could not materialize {len(missing)}/{len(matches_hashes)} statement structures"
            f" (first missing: {preview})"
        )
    return found


def prior_benchmark_pair_sources(exclude: set[Path]) -> dict[tuple[int, int], list[str]]:
    """Map each prior benchmark pair to the files that already contain it."""
    sources: dict[tuple[int, int], set[str]] = defaultdict(set)
    for path in sorted((ROOT / "data/benchmark").glob("*.jsonl")):
        if path.resolve() in exclude or path.name in NONCANONICAL_BENCHMARK_FILES:
            continue
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            key = exact_pair(row, "matches_hash") or exact_pair(row, "pa_hash")
            if key is not None:
                sources[key].add(path.name)
    return {key: sorted(paths) for key, paths in sources.items()}


def resolve_prior_pair_exclusions(
    *,
    pool_path: Path,
    pool_pairs: set[tuple[int, int]],
    discovered_sources: dict[tuple[int, int], list[str]],
) -> dict[tuple[int, int], list[str]]:
    """Keep the canonical pool's exclusions stable as the repository evolves.

    The two known overlaps are part of the frozen default-pool contract.  A new
    benchmark file must not silently remove a pair from an established curation
    prefix, so any newly discovered overlap in that pool is a hard error.  A
    custom pool has no pinned contract and uses its currently discovered
    overlaps.
    """
    discovered_overlaps = pool_pairs & discovered_sources.keys()
    if pool_path.resolve() != DEFAULT_POOL.resolve():
        return {
            pair: discovered_sources[pair]
            for pair in discovered_overlaps
        }

    pinned_pairs = set(PINNED_PRIOR_BENCHMARK_PAIR_SOURCES)
    missing_from_pool = pinned_pairs - pool_pairs
    if missing_from_pool:
        raise ValueError(
            "default representative pool is missing pinned exclusion pairs: "
            f"{sorted(missing_from_pool)}"
        )
    new_overlaps = discovered_overlaps - pinned_pairs
    if new_overlaps:
        raise ValueError(
            "new prior-benchmark overlap would change the canonical selection; "
            f"review and explicitly revise the pinned contract: {sorted(new_overlaps)}"
        )
    return {
        pair: list(source_files)
        for pair, source_files in PINNED_PRIOR_BENCHMARK_PAIR_SOURCES.items()
    }


def validate_curation_tag(row: dict) -> str:
    """Return one canonical INDRA curation tag or fail closed."""
    tag = row.get("tag")
    if not isinstance(tag, str) or tag not in CURATION_TAGS:
        raise ValueError(
            f"curation {row.get('id')!r} has invalid tag {tag!r}; "
            f"expected one of {', '.join(CURATION_TAGS)}"
        )
    return tag


def validate_export_identity(
    *,
    email: str,
    after_id: int,
    benchmark_target: int,
    pool: Path,
    pool_provenance: Path | None,
    dataset_id: str,
    out: Path,
    meta: Path,
    manifest: Path,
    pre_reservoir_manifest: Path,
) -> None:
    """Prevent custom selections from overwriting the canonical ``_400`` files."""
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id must be a non-empty string")
    if (
        isinstance(benchmark_target, bool)
        or not isinstance(benchmark_target, int)
        or benchmark_target < 1
    ):
        raise ValueError("benchmark_target must be a positive integer")

    canonical_identity = (
        email == DEFAULT_CURATOR
        and after_id == DEFAULT_AFTER_ID
        and benchmark_target == BENCHMARK_TARGET_UNIQUE_PAIRS
        and pool.resolve() == DEFAULT_POOL.resolve()
        and pool_provenance is None
        and dataset_id == DEFAULT_DATASET_ID
    )
    default_outputs = {
        DEFAULT_SNAPSHOT.resolve(),
        DEFAULT_META.resolve(),
        DEFAULT_MANIFEST.resolve(),
        DEFAULT_PRE_RESERVOIR_MANIFEST.resolve(),
    }
    requested_outputs = {
        out.resolve(),
        meta.resolve(),
        manifest.resolve(),
        pre_reservoir_manifest.resolve(),
    }
    if len(requested_outputs) != 4:
        raise ValueError("snapshot, metadata, pool manifest, and history manifest outputs must differ")
    if requested_outputs & default_outputs and not canonical_identity:
        raise ValueError(
            "noncanonical curator/cutoff/target/pool/provenance/dataset identity cannot write "
            "canonical representative_indra_curations_400 outputs"
        )
    if not canonical_identity and dataset_id == DEFAULT_DATASET_ID:
        raise ValueError("a noncanonical selection requires a custom --dataset-id")


def eligible_curations(
    universe: list[dict],
    *,
    curator: str,
    after_id: int,
    pool_pairs: set[tuple[int, int]],
    excluded_pairs: set[tuple[int, int]],
    allowed_sources: set[str] | frozenset[str] = ALLOWED_REPRESENTATIVE_CURATION_SOURCES,
) -> list[dict]:
    """Select ordered viewer-lane submissions, never admitting excluded pairs."""
    wanted_curator = curator.strip().lower()
    eligible = []
    for row in universe:
        if (row.get("curator") or "").strip().lower() != wanted_curator:
            continue
        if str(row.get("source") or "") not in allowed_sources:
            continue
        curation_id = exact_integer(row.get("id"))
        if curation_id is None:
            continue
        pair = exact_pair(row, "pa_hash")
        if curation_id > after_id and pair in pool_pairs and pair not in excluded_pairs:
            eligible.append(row)
    eligible.sort(key=lambda row: int(row["id"]))
    return eligible


def select_all_earliest_pairs(
    eligible: list[dict],
) -> tuple[list[dict], list[dict], list[dict], dict[tuple[int, int], list[dict]]]:
    """Retain the earliest event for each pair in curation-id order.

    Input rows are copied into ascending curation-id order and every event is
    scanned, so the result is always the complete current unique-pair snapshot.

    Returns the retained canonical rows, excluded repeat rows, all events
    observed before the cutoff, and those observed events grouped by pair for
    conflict auditing.
    """
    ordered: list[tuple[int, dict]] = []
    seen_ids: set[int] = set()
    for row in eligible:
        curation_id = exact_integer(row.get("id"))
        if curation_id is None:
            raise ValueError("eligible curation is missing an exact integer id")
        if curation_id in seen_ids:
            raise ValueError(f"duplicate curation id {curation_id}")
        seen_ids.add(curation_id)
        pair = exact_pair(row, "pa_hash")
        if pair is None:
            raise ValueError("eligible curation is missing exact pair hashes")
        ordered.append((curation_id, row))
    ordered.sort(key=lambda item: item[0])

    retained: list[dict] = []
    repeats: list[dict] = []
    observed: list[dict] = []
    observed_by_pair: dict[tuple[int, int], list[dict]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()

    for _, row in ordered:
        pair = exact_pair(row, "pa_hash")
        assert pair is not None
        observed.append(row)
        observed_by_pair[pair].append(row)
        if pair in seen:
            repeats.append(row)
        else:
            seen.add(pair)
            retained.append(row)

    return retained, repeats, observed, dict(observed_by_pair)


def pre_reservoir_curations(
    universe: list[dict],
    *,
    curator: str,
    through_id: int,
    source: str = PRE_RESERVOIR_SOURCE,
) -> list[dict]:
    """The genuine old-viewer history, excluding unrelated API/auth probes."""
    wanted_curator = curator.strip().lower()
    rows = []
    for row in universe:
        if (row.get("curator") or "").strip().lower() != wanted_curator:
            continue
        curation_id = exact_integer(row.get("id"))
        pair = exact_pair(row, "pa_hash")
        if (
            curation_id is not None
            and curation_id <= through_id
            and pair is not None
            and str(row.get("source") or "") == source
        ):
            rows.append(row)
    rows.sort(key=lambda row: int(row["id"]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", default=DEFAULT_CURATOR)
    ap.add_argument("--after-id", type=int, default=DEFAULT_AFTER_ID)
    ap.add_argument(
        "--benchmark-target",
        type=int,
        default=BENCHMARK_TARGET_UNIQUE_PAIRS,
        help="benchmark completion target; does not cap the all-current snapshot",
    )
    ap.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    ap.add_argument(
        "--pool-provenance",
        type=Path,
        help="required JSON provenance contract when --pool is not the frozen default",
    )
    ap.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    ap.add_argument("--out", type=Path, default=DEFAULT_SNAPSHOT)
    ap.add_argument("--meta", type=Path, default=DEFAULT_META)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument(
        "--pre-reservoir-manifest",
        type=Path,
        default=DEFAULT_PRE_RESERVOIR_MANIFEST,
    )
    args = ap.parse_args()

    try:
        validate_export_identity(
            email=args.email,
            after_id=args.after_id,
            benchmark_target=args.benchmark_target,
            pool=args.pool,
            pool_provenance=args.pool_provenance,
            dataset_id=args.dataset_id,
            out=args.out,
            meta=args.meta,
            manifest=args.manifest,
            pre_reservoir_manifest=args.pre_reservoir_manifest,
        )
    except ValueError as exc:
        ap.error(str(exc))

    pool_rows, pool = load_pool(args.pool)
    try:
        pool_provenance = load_pool_provenance(args.pool, args.pool_provenance, pool_rows)
    except ValueError as exc:
        ap.error(str(exc))
    discovered_prior_sources = prior_benchmark_pair_sources({
        args.out.resolve(),
        args.manifest.resolve(),
        args.pre_reservoir_manifest.resolve(),
    })
    try:
        prior_sources = resolve_prior_pair_exclusions(
            pool_path=args.pool,
            pool_pairs=set(pool),
            discovered_sources=discovered_prior_sources,
        )
    except ValueError as exc:
        ap.error(str(exc))
    prior_pairs = set(discovered_prior_sources)
    excluded_pool_pairs = set(prior_sources)
    url, key, _ = _load_env()
    universe = pull_all(url, key)
    pre_reservoir_rows = pre_reservoir_curations(
        universe,
        curator=args.email,
        through_id=args.after_id,
    )
    pre_reservoir_grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in pre_reservoir_rows:
        pre_reservoir_grouped[exact_pair(row, "pa_hash")].append(row)
    pre_reservoir_pairs = set(pre_reservoir_grouped)
    eligible = eligible_curations(
        universe,
        curator=args.email,
        after_id=args.after_id,
        pool_pairs=set(pool),
        excluded_pairs=excluded_pool_pairs | pre_reservoir_pairs,
    )
    try:
        selected, repeat_events, observed_events, observed_by_pair = select_all_earliest_pairs(
            eligible
        )
    except ValueError as exc:
        ap.error(str(exc))
    if not selected:
        raise SystemExit(f"no post-{args.after_id} representative unique pairs")
    for row in observed_events:
        validate_curation_tag(row)

    selected_by_pair = {
        exact_pair(row, "pa_hash"): row
        for row in selected
    }
    if len(selected_by_pair) != len(selected):
        raise AssertionError("canonical selection contains a duplicate exact pair")
    if set(selected_by_pair) & (excluded_pool_pairs | pre_reservoir_pairs):
        raise AssertionError("excluded benchmark/history pair entered selected snapshot")

    statement_structures = fetch_statement_structures(
        url,
        {pair[0] for pair in selected_by_pair},
    )

    out_rows = []
    for retained in selected:
        pair = exact_pair(retained, "pa_hash")
        if pair is None:
            raise AssertionError("retained curation lost its exact pair")
        source = pool[pair]
        tag = validate_curation_tag(retained)
        gold = "correct" if is_gold_correct(tag) else "incorrect"
        out_rows.append({
            "matches_hash": pair[0],
            "source_hash": pair[1],
            "source_api": source.get("source_api"),
            "pmid": source.get("pmid"),
            "evidence_text": source.get("text"),
            "stmt_type": statement_structures[pair[0]]["type"],
            "statement": statement_structures[pair[0]],
            "tag": tag,
            "gold": gold,
            "gold_status": "canonical_first_submission",
            "curator": args.email,
            "curation_source": str(retained.get("source") or ""),
            "curation_id": int(retained["id"]),
            "curation_date": retained.get("date"),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.meta.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.pre_reservoir_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in out_rows))

    pre_reservoir_manifest_rows = []
    for pair, rows in sorted(
        pre_reservoir_grouped.items(),
        key=lambda item: min(int(row["id"]) for row in item[1]),
    ):
        rows.sort(key=lambda row: int(row["id"]))
        pre_reservoir_manifest_rows.append({
            "matches_hash": pair[0],
            "source_hash": pair[1],
            "curation_ids": [int(row["id"]) for row in rows],
            "curation_dates": [row.get("date") for row in rows],
            "tags": [str(row.get("tag") or "") for row in rows],
        })
    args.pre_reservoir_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in pre_reservoir_manifest_rows)
    )

    manifest_rows = []
    for row in pool_rows:
        pair = (int(row["stmt_hash"]), int(row["source_hash"]))
        manifest_row = {
            "matches_hash": pair[0],
            "source_hash": pair[1],
            "source_api": row.get("source_api"),
        }
        if pair in prior_sources:
            manifest_row.update({
                "excluded_from_curation": True,
                "exclusion_reason": "preexisting benchmark exact-pair overlap",
                "exclusion_source_files": prior_sources[pair],
            })
        manifest_rows.append(manifest_row)
    args.manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows))

    selected_pairs = set(selected_by_pair)
    pool_pairs = set(pool)
    retained_tags = Counter(str(row.get("tag") or "") for row in selected)
    observed_tags = Counter(str(row.get("tag") or "") for row in observed_events)
    retained_curation_sources = Counter(str(row.get("source") or "") for row in selected)
    observed_curation_sources = Counter(
        str(row.get("source") or "") for row in observed_events
    )
    gold_tags = Counter(row["gold"] for row in out_rows)
    source_mix = Counter(str(row.get("source_api") or "unknown") for row in pool_rows)
    repeat_pair_groups = sum(
        1 for rows in observed_by_pair.values() if len(rows) > 1
    )
    historical_tag_conflicts = sum(
        1
        for rows in observed_by_pair.values()
        if len({row.get("tag") for row in rows}) > 1
    )
    historical_binary_conflicts = sum(
        1 for rows in observed_by_pair.values()
        if {row.get("tag") == "correct" for row in rows} == {True, False}
    )
    excluded_repeat_events = []
    for row in repeat_events:
        pair = exact_pair(row, "pa_hash")
        if pair is None:
            raise AssertionError("repeat curation lost its exact pair")
        retained = selected_by_pair[pair]
        excluded_repeat_events.append({
            "curation_id": int(row["id"]),
            "retained_curation_id": int(retained["id"]),
            "matches_hash": pair[0],
            "source_hash": pair[1],
            "tag": validate_curation_tag(row),
            "curation_date": row.get("date"),
            "curation_source": str(row.get("source") or ""),
        })
    unique_pairs_missing = max(0, args.benchmark_target - len(selected))
    unique_pairs_above_target = max(0, len(selected) - args.benchmark_target)
    benchmark_blockers = {"selection_randomness_unproven": True}
    if unique_pairs_missing:
        benchmark_blockers["unique_pairs_missing"] = unique_pairs_missing
    meta = {
        "schema_version": 2,
        "dataset_id": args.dataset_id,
        "curator": args.email,
        "benchmark_status": "pending",
        "pair_target_status": "complete" if unique_pairs_missing == 0 else "incomplete",
        "benchmark_target": {
            "unit": "unique exact (statement, evidence) pairs",
            "unique_pairs": args.benchmark_target,
        },
        "benchmark_blockers": benchmark_blockers,
        "snapshot_semantics": (
            "earliest qualifying submission for each exact pair across all qualifying "
            "events returned at export time; benchmark target does not cap the snapshot"
        ),
        "selection_auditability": {
            "reservoir_membership_proven": True,
            "historical_draw_log_available": False,
            "historical_sampler_allowed_replacement": True,
            "simple_random_completed_subset_proven": False,
            "reason": (
                "the historical viewer retained no draw/skip log and retried "
                "unmaterializable or textless rows"
            ),
        },
        "pre_reservoir_curator_history": {
            "through_curation_id": args.after_id,
            "source_filter": PRE_RESERVOIR_SOURCE,
            "raw_submissions": len(pre_reservoir_rows),
            "unique_pairs": len(pre_reservoir_pairs),
            "selected_pair_overlap": len(selected_pairs & pre_reservoir_pairs),
            "manifest_path": path_label(args.pre_reservoir_manifest),
            "manifest_sha256": sha256(args.pre_reservoir_manifest),
        },
        "source": "INDRA DB keyed /curation/list exact-joined to the viewer representative pool",
        "statement_materialization": {
            "endpoint": "/statements/from_hashes",
            "format": "json-js",
            "ev_limit": 1,
            "unique_statements": len(statement_structures),
            "removed_fields": sorted(STATEMENT_FIELDS_REMOVED),
        },
        "selection": {
            "after_curation_id": args.after_id,
            "scope": "all qualifying events returned at export time",
            "canonicalization": "earliest qualifying submission per exact pair",
            "repeat_handling": "exclude later submissions for an already-retained exact pair",
            "allowed_curation_sources": sorted(ALLOWED_REPRESENTATIVE_CURATION_SOURCES),
            "first_curation_id": min(int(row["id"]) for row in selected),
            "last_curation_id": max(int(row["id"]) for row in selected),
            "first_curation_date": selected[0].get("date"),
            "last_curation_date": selected[-1].get("date"),
            "last_observed_event_id": max(int(row["id"]) for row in observed_events),
            "last_observed_event_date": observed_events[-1].get("date"),
        },
        "counts": {
            "observed_submission_events": len(observed_events),
            "retained_submission_events": len(selected),
            "excluded_repeat_submission_events": len(repeat_events),
            "unique_pairs": len(selected),
            "unique_pairs_remaining_to_benchmark_target": unique_pairs_missing,
            "unique_pairs_above_benchmark_target": unique_pairs_above_target,
            "unique_statements": len({pair[0] for pair in selected_by_pair}),
            "gold_correct": gold_tags["correct"],
            "gold_incorrect": gold_tags["incorrect"],
        },
        "deduplication_audit": {
            "identity": (
                "exact artifact (matches_hash, source_hash) pair; "
                "INDRA API input aliases matches_hash as pa_hash"
            ),
            "rule": "first qualifying submission wins",
            "repeat_pair_groups": repeat_pair_groups,
            "historical_tag_conflict_pairs": historical_tag_conflicts,
            "historical_binary_conflict_pairs": historical_binary_conflicts,
            "excluded_repeat_events": excluded_repeat_events,
            "observed_event_tag_counts": dict(sorted(observed_tags.items())),
            "observed_event_curation_source_counts": dict(
                sorted(observed_curation_sources.items())
            ),
        },
        "retained_tag_counts": dict(sorted(retained_tags.items())),
        "retained_curation_source_counts": dict(
            sorted(retained_curation_sources.items())
        ),
        "contamination": {
            "selected_pairs_overlapping_preexisting_benchmarks": len(selected_pairs & prior_pairs),
            "pool_pairs_overlapping_preexisting_benchmarks": len(pool_pairs & prior_pairs),
        },
        "representative_pool": {
            "algorithm": pool_provenance["algorithm"],
            "sampling_unit": pool_provenance["sampling_unit"],
            "without_replacement": pool_provenance["without_replacement"],
            "seed": pool_provenance["seed"],
            "population_rows": pool_provenance["population_rows"],
            "source_dump": pool_provenance["source_dump"],
            "source_dump_sha256": pool_provenance["source_dump_sha256"],
            "materialization_path": path_label(args.pool),
            "materialization_sha256": pool_provenance["materialization_sha256"],
            "manifest_path": path_label(args.manifest),
            "manifest_sha256": sha256(args.manifest),
            "pairs": len(pool),
            "statements": len({pair[0] for pair in pool}),
            "source_api_counts": dict(sorted(source_mix.items())),
        },
        "artifact": {
            "path": path_label(args.out),
            "sha256": sha256(args.out),
        },
    }
    args.meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(
        f"retained {len(out_rows)} unique pairs from {len(observed_events)} observed submissions; "
        f"excluded {len(repeat_events)} later repeats "
        f"({gold_tags['correct']} correct / {gold_tags['incorrect']} incorrect)"
    )
    print(
        f"wrote {args.out}, {args.meta}, {args.manifest}, "
        f"and {args.pre_reservoir_manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
