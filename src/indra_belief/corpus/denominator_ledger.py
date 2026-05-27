"""Unified denominator ledger (B1 of deferred hypergraph).

Reads the `denominator_ledger` VIEW defined in `corpus/schema.py`. Each row
names a `(family, kind)` and a count; `run_id` is set for run-scoped rows
and NULL for corpus-level totals.

Consumers (UI cells, model cards, preflight previews) should fetch by
filter, e.g. `query_denominator_ledger(con, run_id=R, family='aggregate')`.
A future Phase will expose this in the SvelteKit viewer so denominator
cells can link to the producing ledger row.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import duckdb


@dataclass(frozen=True)
class DenominatorLedgerRow:
    run_id: Optional[str]
    family: str
    kind: str
    value: int
    slice_json: Optional[str]


def query_denominator_ledger(
    con: "duckdb.DuckDBPyConnection",
    *,
    run_id: Optional[str] = None,
    family: Optional[str] = None,
    kind: Optional[str] = None,
) -> list[DenominatorLedgerRow]:
    """Query the denominator ledger view with optional filters.

    Passing `run_id=None` returns both run-scoped and corpus-level rows.
    Pass an explicit run_id (or pass empty string `""` if you ever need
    only the NULL-run-id corpus rows — though the VIEW emits NULL not "",
    so this query uses `IS NULL` semantics for the no-filter case).
    """
    where = []
    params: list[object] = []
    if run_id is not None:
        where.append("run_id = ?")
        params.append(run_id)
    if family is not None:
        where.append("family = ?")
        params.append(family)
    if kind is not None:
        where.append("kind = ?")
        params.append(kind)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    rows = con.execute(
        f"""
        SELECT run_id, family, kind, value, CAST(slice_json AS VARCHAR) AS slice_json
          FROM denominator_ledger
        {where_sql}
         ORDER BY run_id NULLS FIRST, family, kind
        """,
        params,
    ).fetchall()
    return [
        DenominatorLedgerRow(
            run_id=row[0],
            family=str(row[1]),
            kind=str(row[2]),
            value=int(row[3]),
            slice_json=(str(row[4]) if row[4] is not None else None),
        )
        for row in rows
    ]
