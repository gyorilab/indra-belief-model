"""Canonical content-address spine for the comparison pipeline.

This is the single home for the strict canonical JSON codec and the sha256
primitives every consumer imports. It is deliberately dependency-free so that
`contracts.py`, `assemble.py`, and `metrics.py` all share ONE byte-exact
implementation — the producer/consumer digest contract is then structural, not
a copy-paste coincidence.

The encoder here is ensure_ascii=False (raw UTF-8). The on-disk spend-ledger
encoder in `spend_guard._ledger_json_bytes` is a SEPARATE, ensure_ascii=True
codec and is intentionally not routed through here (see that module).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


class HashingError(ValueError):
    """Raised when a value cannot be canonicalized or a file cannot be hashed.

    Subclasses ValueError so existing broad handlers keep absorbing it on the
    (defensive, test-uncovered) failure paths that previously raised
    ContractError / AssemblyError.
    """


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise HashingError("value is not strict canonical JSON") from exc


def canonical_json_line(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise HashingError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def ordered_statement_id_sha256(statement_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for statement_id in statement_ids:
        digest.update(
            json.dumps(
                {"statement_id": statement_id},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()
