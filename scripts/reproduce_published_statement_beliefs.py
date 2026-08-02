#!/usr/bin/env python3
"""Re-derive every published LLM statement belief from its frozen observations.

The four published comparison arms ship two prediction files each — an
all-source panel (1689 statements) and a five-reader panel (1676) — for 13460
scores in total. Those numbers were produced by ``statement_belief`` reading
per-evidence verdicts off a raw attempts log. This script rebuilds the join
from the frozen inputs each bundle's own manifest names, re-runs the live
aggregator over it, and compares against the shipped files at EXACT float
equality. A single nonzero delta exits 1 naming the arm, panel, and statement.

That is the behavioural freeze. The manifests also record the sha256 of
``statement_belief.py`` itself, and a byte digest is the cheaper freeze — but
it forbids even a comment. This one forbids only a change in what the file
computes, which is the property the published numbers actually depend on.

Every input is verified against the sha256 and byte count its manifest
declares before it is read, so "frozen observations" means the bytes the
bundle was built from, not whatever happens to sit at that path today.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/reproduce_published_statement_beliefs.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.comparison.llm import (  # noqa: E402
    READER_SOURCES,
    _aggregation_config,
)
from indra_belief.statement_belief import statement_belief  # noqa: E402

MODELS_DIR = ROOT / "data" / "comparison" / "models"

# Fixed presentation order, matching the framing artifact's arm order.
PUBLISHED_ARMS = ("gemma_4_26b", "glm_5", "gemma_4_31b", "gemma_4_e2b")

# The census the bundles declare; a drift here is a substrate change, not a
# rounding difference, and should fail loudly rather than silently rescope.
EXPECTED_STATEMENTS = 1689
EXPECTED_EXECUTIONS = 33361
EXPECTED_READER_STATEMENTS = 1676
EXPECTED_SCORES = len(PUBLISHED_ARMS) * (EXPECTED_STATEMENTS + EXPECTED_READER_STATEMENTS)

PANELS = ("all_source", "reader")
_PANEL_FILES = {
    "all_source": "all_source_predictions.jsonl",
    "reader": "reader_predictions.jsonl",
}


class ReproductionError(RuntimeError):
    """A frozen input is missing, altered, or shaped unlike its manifest."""


@dataclass(frozen=True)
class Mismatch:
    arm: str
    panel: str
    statement_id: str
    published: float | None
    rederived: float | None

    @property
    def delta(self) -> float:
        if self.published is None or self.rederived is None:
            return float("inf")
        return abs(self.published - self.rederived)

    def describe(self) -> str:
        return (
            f"{self.arm}/{self.panel}/{self.statement_id}: "
            f"published={self.published!r} rederived={self.rederived!r} "
            f"delta={self.delta!r}"
        )


@dataclass(frozen=True)
class Report:
    scores: int
    files: int
    mismatches: tuple[Mismatch, ...]
    seconds: float

    @property
    def max_delta(self) -> float:
        return max((m.delta for m in self.mismatches), default=0.0)

    @property
    def ok(self) -> bool:
        return not self.mismatches and self.scores == EXPECTED_SCORES


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_verified(path: Path, descriptor: dict[str, Any], label: str) -> bytes:
    """Read a small frozen input, refusing bytes its manifest does not declare."""
    payload = path.read_bytes()
    _check_descriptor(path, descriptor, label, len(payload), _sha256_bytes(payload))
    return payload


def _check_descriptor(
    path: Path,
    descriptor: dict[str, Any],
    label: str,
    size: int,
    digest: str,
) -> None:
    if size != descriptor["bytes"]:
        raise ReproductionError(
            f"{label} at {path} is {size} bytes; manifest declares "
            f"{descriptor['bytes']}"
        )
    if digest != descriptor["sha256"]:
        raise ReproductionError(
            f"{label} at {path} hashes to {digest}; manifest declares "
            f"{descriptor['sha256']}"
        )


def _resolve(manifest_path: Path, descriptor: dict[str, Any]) -> Path:
    return (manifest_path.parent / descriptor["path"]).resolve()


def _jsonl(payload: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in payload.splitlines() if line.strip()]


def _final_measurements(
    path: Path, descriptor: dict[str, Any], label: str
) -> dict[tuple[int, int], tuple[Any, Any, Any]]:
    """Stream a multi-GB attempts log once, hashing and reducing in one pass.

    Retries precede their successor for the same pair, so the LAST row for a
    key is the final one — the same ``rows[-1]`` rule ``llm._validate_raw``
    applies when it builds the published bundle.
    """
    digest = hashlib.sha256()
    size = 0
    final: dict[tuple[int, int], tuple[Any, Any, Any]] = {}
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            size += len(raw_line)
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            key = (row["stmt_i"], row["evidence_i"])
            final[key] = (row.get("verdict"), row.get("confidence"), row.get("tier"))
    _check_descriptor(path, descriptor, label, size, digest.hexdigest())
    return final


def _statement_ids(statements: list[dict[str, Any]]) -> list[str]:
    ids = [statement["id"] for statement in statements]
    if len(set(ids)) != len(ids):
        raise ReproductionError("statement IDs repeat in the frozen corpus")
    return ids


def _panels(
    *,
    statements: list[dict[str, Any]],
    map_rows: list[dict[str, Any]],
    final: dict[tuple[int, int], tuple[Any, Any, Any]],
    priors: dict[str, tuple[float, float]],
) -> dict[str, list[dict[str, Any]]]:
    """Rebuild both published panels for one arm.

    Mirrors ``llm._predictions``: pairs in (stmt_i, evidence_i) order, the
    five-reader panel keeping only READER_SOURCES rows and omitting statements
    left with nothing.
    """
    ids = _statement_ids(statements)
    all_evidence: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(ids))}
    reader_evidence: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(ids))}

    keys = sorted((row["new_stmt_i"], row["new_evidence_i"]) for row in map_rows)
    if len(keys) != len(set(keys)):
        raise ReproductionError("execution map repeats a statement/evidence key")
    by_key = {(row["new_stmt_i"], row["new_evidence_i"]): row for row in map_rows}
    missing = [key for key in keys if key not in final]
    if missing:
        raise ReproductionError(
            f"raw attempts omit {len(missing)} execution-map pairs, "
            f"first {missing[0]}"
        )

    for key in keys:
        stmt_i, evidence_i = key
        map_row = by_key[key]
        evidence = statements[stmt_i]["evidence"][evidence_i]
        verdict, confidence, tier = final[key]
        source = str(evidence.get("source_api") or "").casefold()
        measurement = {
            "source_api": source,
            "verdict": verdict,
            "confidence": confidence,
            "tier": tier,
            "evidence_text": evidence.get("text"),
            "evidence_hash": map_row["evidence_json_sha256"],
        }
        all_evidence[stmt_i].append(measurement)
        if source in READER_SOURCES:
            reader_evidence[stmt_i].append(measurement)

    def score(rows: list[dict[str, Any]]) -> float:
        belief = statement_belief(rows, priors=priors, dedup=True, soft=None).belief
        if belief is None:
            raise ReproductionError("a published panel statement has no belief")
        return float(belief)

    return {
        "all_source": [
            {"probability_correct": score(all_evidence[i]), "statement_id": ids[i]}
            for i in range(len(ids))
        ],
        "reader": [
            {"probability_correct": score(reader_evidence[i]), "statement_id": ids[i]}
            for i in range(len(ids))
            if reader_evidence[i]
        ],
    }


def _compare(
    arm: str, panel: str, published: list[dict[str, Any]], rederived: list[dict[str, Any]]
) -> list[Mismatch]:
    """Compare keyed on statement_id, at exact float equality — never approx."""
    mismatches: list[Mismatch] = []
    published_by_id = {row["statement_id"]: row["probability_correct"] for row in published}
    rederived_by_id = {row["statement_id"]: row["probability_correct"] for row in rederived}
    if [row["statement_id"] for row in published] != [
        row["statement_id"] for row in rederived
    ]:
        mismatches.append(Mismatch(arm, panel, "<row order>", None, None))
    for statement_id in sorted(set(published_by_id) | set(rederived_by_id)):
        theirs = published_by_id.get(statement_id)
        ours = rederived_by_id.get(statement_id)
        if theirs is None or ours is None or theirs != ours:
            mismatches.append(Mismatch(arm, panel, statement_id, theirs, ours))
    return mismatches


def reproduce(arms: tuple[str, ...] = PUBLISHED_ARMS, *, verbose: bool = False) -> Report:
    """Re-derive and compare every published prediction file. Never mutates disk."""
    started = time.monotonic()
    corpus_cache: dict[str, Any] = {}
    mismatches: list[Mismatch] = []
    scores = 0
    files = 0

    for arm in arms:
        manifest_path = MODELS_DIR / arm / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        notes = manifest["implementation"]["notes"]
        if notes["aggregation"] != "indra_default_hard_gate" or notes["reader_profile"] is not None:
            raise ReproductionError(
                f"{arm} is not the published hard-gate/unfitted configuration"
            )
        inputs = notes["inputs"]

        aggregation_descriptor = inputs["aggregation_config"]
        cache_key = aggregation_descriptor["sha256"]
        if cache_key not in corpus_cache:
            raw = _read_verified(
                _resolve(manifest_path, aggregation_descriptor),
                aggregation_descriptor,
                f"{arm} aggregation_config",
            )
            raw_priors, profile, _name = _aggregation_config(json.loads(raw))
            if profile is not None:
                raise ReproductionError("published aggregation declares a reader profile")
            corpus_cache[cache_key] = {
                key: (float(value[0]), float(value[1]))
                for key, value in raw_priors.items()
            }
        priors = corpus_cache[cache_key]

        statements_descriptor = inputs["statements"]
        cache_key = statements_descriptor["sha256"]
        if cache_key not in corpus_cache:
            statements = json.loads(
                _read_verified(
                    _resolve(manifest_path, statements_descriptor),
                    statements_descriptor,
                    f"{arm} statements",
                )
            )
            if len(statements) != EXPECTED_STATEMENTS:
                raise ReproductionError(
                    f"{arm} statement corpus holds {len(statements)}, "
                    f"expected {EXPECTED_STATEMENTS}"
                )
            corpus_cache[cache_key] = statements
        statements = corpus_cache[cache_key]

        map_descriptor = inputs["execution_map"]
        cache_key = map_descriptor["sha256"]
        if cache_key not in corpus_cache:
            map_rows = _jsonl(
                _read_verified(
                    _resolve(manifest_path, map_descriptor),
                    map_descriptor,
                    f"{arm} execution_map",
                )
            )
            if len(map_rows) != EXPECTED_EXECUTIONS:
                raise ReproductionError(
                    f"{arm} execution map holds {len(map_rows)} rows, "
                    f"expected {EXPECTED_EXECUTIONS}"
                )
            corpus_cache[cache_key] = map_rows
        map_rows = corpus_cache[cache_key]

        raw_descriptor = inputs["raw_attempts"]
        final = _final_measurements(
            _resolve(manifest_path, raw_descriptor),
            raw_descriptor,
            f"{arm} raw_attempts",
        )

        panels = _panels(
            statements=statements, map_rows=map_rows, final=final, priors=priors
        )
        if len(panels["reader"]) != EXPECTED_READER_STATEMENTS:
            raise ReproductionError(
                f"{arm} five-reader panel holds {len(panels['reader'])} statements, "
                f"expected {EXPECTED_READER_STATEMENTS}"
            )

        for panel in PANELS:
            published = _jsonl((MODELS_DIR / arm / _PANEL_FILES[panel]).read_bytes())
            arm_mismatches = _compare(arm, panel, published, panels[panel])
            mismatches.extend(arm_mismatches)
            scores += len(panels[panel])
            files += 1
            if verbose:
                status = "OK" if not arm_mismatches else f"{len(arm_mismatches)} MISMATCH"
                print(f"  {arm:<12} {panel:<10} {len(panels[panel]):>5} scores  {status}")

    return Report(
        scores=scores,
        files=files,
        mismatches=tuple(mismatches),
        seconds=time.monotonic() - started,
    )


@lru_cache(maxsize=1)
def published_reproduction() -> Report:
    """The full reproduction, computed once per process.

    ``reproduce`` is a pure read over bytes pinned by manifest sha256, so the
    several contract tests that lean on the behavioural freeze can share one
    ~35s run instead of each paying for it.
    """
    return reproduce(PUBLISHED_ARMS)


def main(argv: list[str]) -> int:
    arms = tuple(argv[1:]) or PUBLISHED_ARMS
    unknown = [arm for arm in arms if arm not in PUBLISHED_ARMS]
    if unknown:
        print(f"unknown arm(s): {unknown}", file=sys.stderr)
        return 2
    try:
        report = reproduce(arms, verbose=True)
    except ReproductionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        f"{report.files} files, {report.scores} scores, "
        f"{len(report.mismatches)} mismatches, max delta {report.max_delta!r}, "
        f"{report.seconds:.1f}s"
    )
    if report.mismatches:
        for mismatch in report.mismatches[:20]:
            print(f"FAIL: {mismatch.describe()}", file=sys.stderr)
        if len(report.mismatches) > 20:
            print(f"FAIL: ... {len(report.mismatches) - 20} more", file=sys.stderr)
        return 1
    if report.scores != EXPECTED_SCORES:
        print(
            f"FAIL: scored {report.scores}, expected {EXPECTED_SCORES}", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
