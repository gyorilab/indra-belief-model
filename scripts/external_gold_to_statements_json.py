"""Convert data/benchmark/external_gold_v1.jsonl into a statements JSON the
monolithic runner can score, ONE single-evidence INDRA statement per gold row,
in FILE ORDER (output stmt_i == input line index == gold row index — the join
the eval depends on).

Why not eval_to_statements_json.py: that path needs a CorpusIndex and only
resolves 10/154 of these rows. external_gold_v1 carries the subject/object/
evidence inline, so we reconstruct statements directly from those fields.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/external_gold_to_statements_json.py \
    [--input data/benchmark/external_gold_v1.jsonl] \
    [--output data/corpora/external_gold_v1_statements.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from indra.statements import (  # noqa: E402
    Activation,
    Agent,
    Complex,
    DecreaseAmount,
    Dephosphorylation,
    Evidence,
    IncreaseAmount,
    Inhibition,
    Phosphorylation,
)

DEFAULT_INPUT = ROOT / "data" / "benchmark" / "external_gold_v1.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "corpora" / "external_gold_v1_statements.json"
DEFAULT_SIDECAR = ROOT / "data" / "corpora" / "external_gold_v1_statements.index.json"


def build_statement(row: dict):
    """One single-evidence statement from a gold row's inline fields.

    subj/obj carry only TEXT db_refs so grounding re-resolves them at score
    time (the substrate-repair point). Evidence carries text/source_api/pmid;
    no source_hash kwarg exists on Evidence and _from_json drops it anyway —
    the runner regenerates source_hash, so the join is on stmt_i, not hash.
    """
    subj = Agent(row["subject"], db_refs={"TEXT": row["subject"]})
    obj = Agent(row["object"], db_refs={"TEXT": row["object"]})
    ev = Evidence(
        text=row["evidence_text"],
        source_api=row.get("source_api"),
        pmid=row.get("pmid"),
    )
    st = row["stmt_type"]
    if st == "Complex":
        return Complex([subj, obj], evidence=[ev])
    if st == "Phosphorylation":
        return Phosphorylation(subj, obj, evidence=[ev])
    if st == "Dephosphorylation":
        return Dephosphorylation(subj, obj, evidence=[ev])
    if st == "Activation":
        return Activation(subj, obj, evidence=[ev])
    if st == "Inhibition":
        return Inhibition(subj, obj, evidence=[ev])
    if st == "IncreaseAmount":
        return IncreaseAmount(subj, obj, evidence=[ev])
    if st == "DecreaseAmount":
        return DecreaseAmount(subj, obj, evidence=[ev])
    raise ValueError(f"unsupported stmt_type: {st!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--sidecar", default=str(DEFAULT_SIDECAR))
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    stmts = []
    failures = []
    sidecar = []
    for i, row in enumerate(rows):
        try:
            st = build_statement(row)
        except Exception as exc:  # noqa: BLE001
            failures.append((i, f"{type(exc).__name__}: {exc}"))
            continue
        stmts.append(st)
        sidecar.append({
            "output_index": len(stmts) - 1,
            "gold_row_index": i,
            "source_hash": row.get("source_hash"),
            "gold": row.get("gold"),
            "stmt_type": row.get("stmt_type"),
        })

    if failures:
        print(f"FAILED to build {len(failures)} rows:", file=sys.stderr)
        for i, why in failures:
            print(f"  row {i}: {why}", file=sys.stderr)

    out = [s.to_json() for s in stmts]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out))
    Path(args.sidecar).write_text(json.dumps(sidecar, indent=2))

    # output stmt_i must equal gold row index: only true if zero failures
    # (a skipped row would shift every subsequent index).
    assert len(out) == len(rows) - len(failures)
    if not failures:
        assert len(out) == 154, f"expected 154, got {len(out)}"

    print(
        f"built {len(out)}/{len(rows)} statements -> {args.output}\n"
        f"sidecar -> {args.sidecar}\n"
        f"failures: {len(failures)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
