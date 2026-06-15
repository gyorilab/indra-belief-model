"""Shared helpers for the holdout error-profile / comparison analysis scripts.

DRY lift (R4): these bodies were duplicated byte-for-byte (modulo whitespace and
type-annotation noise) across y_phase_error_profile, z_phase_error_profile,
cc_holdout_cc_compare, and three_way_holdout_cc. Behavior is unchanged — every
caller passes the same argument shapes it did before, and the canonical
``emit_table`` coerces header cells with ``str()`` (a no-op on the string-literal
headers the y/z callers pass).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).parent.parent


def load_source(name: str) -> dict:
    src: dict = {}
    p = ROOT / "data" / "benchmark" / f"{name}.jsonl"
    with open(p) as fh:
        for line in fh:
            r = json.loads(line)
            src[r["source_hash"]] = r
    return src


def load_run(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def mcnemar_exact_pvalue(b: int, c: int) -> float:
    """Exact (mid-p) McNemar test on discordant pairs.

    H_0: P(AA-correct, CC-wrong) = P(AA-wrong, CC-correct).
    Two-sided p-value: 2 × P(X ≥ max(b,c)) under Binomial(b+c, 0.5).
    Returns 1.0 when b == c == 0.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    # Right-tail probability under Binom(n, 0.5)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(2 * tail, 1.0)


def emit_section(out, title: str) -> None:
    out.write(f"\n## {title}\n\n")


def emit_table(out, header: list, rows: list) -> None:
    out.write("| " + " | ".join(str(c) for c in header) + " |\n")
    out.write("|" + "|".join(["---"] * len(header)) + "|\n")
    for row in rows:
        out.write("| " + " | ".join(str(c) for c in row) + " |\n")
    out.write("\n")
