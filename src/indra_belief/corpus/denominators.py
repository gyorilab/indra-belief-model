"""Evidence-denominator validation for corpus statement rows.

The viewer treats statement raw JSON and the normalized evidence table as two
views of the same scorable population. If they diverge, cost previews and model
exports can describe a different corpus than the worker will actually load.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import duckdb


@dataclass(frozen=True)
class EvidenceDenominatorMismatch:
    stmt_hash: str
    raw_json_evidences: int
    table_evidences: int


@dataclass(frozen=True)
class EvidenceDenominatorValidation:
    n_statements: int
    n_raw_json_evidences: int
    n_table_evidences: int
    mismatches: tuple[EvidenceDenominatorMismatch, ...]

    @property
    def validated(self) -> bool:
        return not self.mismatches

    def to_dict(self) -> dict:
        return {
            "n_statements": self.n_statements,
            "n_raw_json_evidences": self.n_raw_json_evidences,
            "n_table_evidences": self.n_table_evidences,
            "evidence_count_validated": self.validated,
            "mismatches": [
                {
                    "stmt_hash": m.stmt_hash,
                    "n_raw_json_evidences": m.raw_json_evidences,
                    "n_table_evidences": m.table_evidences,
                }
                for m in self.mismatches
            ],
        }


def _table_exists(con: "duckdb.DuckDBPyConnection", table: str) -> bool:
    row = con.execute(
        """
        SELECT COUNT(*)
          FROM information_schema.tables
         WHERE table_name = ?
        """,
        [table],
    ).fetchone()
    return bool(row and row[0])


def raw_json_evidence_count(stmt_hash: str, raw_json: str) -> int:
    try:
        parsed = json.loads(raw_json)
    except Exception as err:
        raise ValueError(
            f"statement {stmt_hash} raw_json is not parseable JSON: {err}"
        ) from err
    if not isinstance(parsed, dict):
        raise ValueError(f"statement {stmt_hash} raw_json is not an object")
    evidence = parsed.get("evidence")
    if evidence is None:
        return 0
    if not isinstance(evidence, list):
        raise ValueError(f"statement {stmt_hash} raw_json has no evidence array")
    return len(evidence)


def validate_statement_evidence_denominators(
    con: "duckdb.DuckDBPyConnection",
    stmt_hashes: Iterable[str] | None = None,
    *,
    raise_on_mismatch: bool = True,
) -> EvidenceDenominatorValidation:
    """Compare `statement.raw_json.evidence[]` counts with evidence rows.

    `stmt_hashes=None` validates the whole corpus. Supplying a hash iterable
    validates exactly those statement rows and raises if any requested row is
    missing, because a missing statement cannot prove denominator agreement.
    """
    params: list[str] = []
    where = ""
    requested: set[str] | None = None
    if stmt_hashes is not None:
        requested = {str(stmt_hash) for stmt_hash in stmt_hashes}
        if not requested:
            return EvidenceDenominatorValidation(0, 0, 0, ())
        params = sorted(requested)
        where = f"WHERE s.stmt_hash IN ({','.join(['?'] * len(params))})"

    rows = con.execute(
        f"""
        SELECT
            s.stmt_hash,
            CAST(s.raw_json AS VARCHAR) AS raw_json,
            COUNT(se.evidence_hash) AS n_table_evidences
        FROM statement s
        LEFT JOIN statement_evidence se ON se.stmt_hash = s.stmt_hash
        {where}
        GROUP BY s.stmt_hash, s.raw_json
        ORDER BY s.stmt_hash
        """,
        params,
    ).fetchall()

    seen = {str(row[0]) for row in rows}
    if requested is not None:
        missing = sorted(requested - seen)
        if missing:
            raise ValueError(
                "statement evidence denominator validation missing statement "
                f"row(s): {', '.join(missing[:5])}"
            )

    n_raw = 0
    n_table = 0
    mismatches: list[EvidenceDenominatorMismatch] = []
    for stmt_hash, raw_json, table_count in rows:
        stmt_hash = str(stmt_hash)
        raw_count = raw_json_evidence_count(stmt_hash, str(raw_json))
        table_count = int(table_count or 0)
        n_raw += raw_count
        n_table += table_count
        if raw_count != table_count:
            mismatches.append(
                EvidenceDenominatorMismatch(
                    stmt_hash=stmt_hash,
                    raw_json_evidences=raw_count,
                    table_evidences=table_count,
                )
            )

    validation = EvidenceDenominatorValidation(
        n_statements=len(rows),
        n_raw_json_evidences=n_raw,
        n_table_evidences=n_table,
        mismatches=tuple(mismatches),
    )
    if raise_on_mismatch and mismatches:
        first = mismatches[0]
        raise ValueError(
            "statement evidence denominator mismatch for "
            f"{first.stmt_hash}: raw_json has {first.raw_json_evidences} "
            f"evidence row{'s' if first.raw_json_evidences != 1 else ''}, "
            f"normalized evidence rows have {first.table_evidences}"
        )
    return validation
