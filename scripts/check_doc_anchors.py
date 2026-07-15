"""Guard the live task-hypergraph docs against file-anchor drift (node R6).

The consistency-reconciliation audit found docs hardcoding `src/`/`scripts/`
paths (often with a `:line` suffix) that the code had since moved or deleted —
e.g. the removed `src/indra_belief/composed_scorer.py`, still cited in the
calibration doc long after it was gone. Nothing failed CI when an anchor went
stale, so drift accumulated silently. This guard scans the live hypergraph
docs, extracts every referenced `src/...py` / `scripts/...py` path, and asserts
each one still exists on disk. It is wired into pytest via
tests/test_doc_anchors.py, mirroring how scripts/check_contamination.py is wired
via tests/test_contamination_guard_sources.py.

Scope:
  * Scans the LIVE docs only (never research/archive/**):
      research/learnings_task_hypergraph.md
      research/calibration_task_hypergraph.md
      research/belief_instrument_task_graph.md
  * Extracts backticked-or-bare repo paths of the form `src/....py` /
    `scripts/....py`, with or without a trailing `:line` / `:a-b` anchor
    (the extractor stops at the `.py`, so the line suffix is ignored).
  * A referenced file that does not exist under the repo root is a DEAD ANCHOR.

What is deliberately NOT treated as a dead anchor (neither is drift):
  * A line in an explicitly SUPERSEDED / historical context — either the line
    itself carries "(superseded)"/"(historical)"/"SUPERSEDED", or it sits under
    a Markdown heading whose text contains "superseded"/"historical" (until the
    next heading). Kept to those explicit markers so live anchors stay visible.
  * A path the docs explicitly say to BUILD — a to-be-created artifact, e.g.
    "**DO.** Build `scripts/probe_effective_context.py`". A planned file is a
    plan for future work, not a stale reference to something that used to exist.

Exit code 0 = every live anchor resolves, 1 = one or more dead anchors.

Usage:
    python scripts/check_doc_anchors.py [DOC ...]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Live task-hypergraph docs to scan (never research/archive/**).
DOCS = [
    ROOT / "research" / "learnings_task_hypergraph.md",
    ROOT / "research" / "calibration_task_hypergraph.md",
    ROOT / "research" / "belief_instrument_task_graph.md",
]

# A repo path to a Python file under src/ or scripts/. The character class stops
# at the ".py" extension, so a trailing ":48-49" / ":16" line-anchor (not in the
# class) is naturally excluded from the captured path.
_PATH = re.compile(r"(?:src|scripts)/[A-Za-z0-9_./-]+\.py")

# Explicit "this block is not current" markers.
_SUPERSEDED_INLINE = re.compile(r"\(superseded\)|\(historical\)|SUPERSEDED",
                                re.IGNORECASE)
_SUPERSEDED_HEADING = re.compile(r"superseded|historical", re.IGNORECASE)

# A line that declares a path as a to-be-built artifact (a plan, not an anchor),
# e.g. "**DO.** Build `scripts/probe_effective_context.py`; run; ..." or a
# "**Targets.** new script" convention.
_BUILD_INTENT = re.compile(r"\b(build|create|scaffold)\b|new script|new file",
                           re.IGNORECASE)


def _rel(p: Path) -> str:
    """Repo-relative string for reporting; falls back to the raw path for docs
    that live outside the repo root (e.g. a pytest tmp_path fixture)."""
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def _is_heading(line: str) -> bool:
    return line.lstrip().startswith("#")


def _planned_paths(records: list[tuple[str, int, str, bool]]) -> set[str]:
    """Paths the docs explicitly say to BUILD — treated as plans, not anchors.

    A path is 'planned' if any line referencing it also carries a build intent
    ('Build `x`', 'new script', ...). Collected across ALL lines so that a later
    bare reference to the same planned file (e.g. an '**Artifacts.**' line) is
    not mistaken for a dead anchor.
    """
    planned: set[str] = set()
    for _doc, _lineno, text, _skipped in records:
        if _BUILD_INTENT.search(text):
            for m in _PATH.finditer(text):
                planned.add(m.group(0))
    return planned


def find_dead_anchors(paths: list | None = None) -> list[dict]:
    """Return dead-anchor records for the given docs (defaults to DOCS).

    Each record is {"doc": str (repo-relative), "line": int, "path": str}. An
    empty list means every live anchor resolves. Importable so the pytest guard
    can call it directly.
    """
    if paths is None:
        paths = DOCS

    records: list[tuple[str, int, str, bool]] = []  # (doc, lineno, text, skip)
    misses: list[dict] = []
    for p in paths:
        p = Path(p)
        doc_rel = _rel(p)
        if not p.exists():
            # A missing DOC is itself a drift signal — report it explicitly.
            misses.append({"doc": doc_rel, "line": 0, "path": "<doc not found>"})
            continue
        heading_superseded = False
        for i, raw in enumerate(p.read_text().splitlines(), 1):
            if _is_heading(raw):
                heading_superseded = bool(_SUPERSEDED_HEADING.search(raw))
            skipped = heading_superseded or bool(_SUPERSEDED_INLINE.search(raw))
            records.append((doc_rel, i, raw, skipped))

    planned = _planned_paths(records)

    for doc_rel, lineno, text, skipped in records:
        if skipped:
            continue
        for m in _PATH.finditer(text):
            rel = m.group(0)
            if rel in planned:
                continue
            if not (ROOT / rel).exists():
                misses.append({"doc": doc_rel, "line": lineno, "path": rel})
    return misses


def main(argv: list | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    docs = [Path(a) for a in argv] if argv else DOCS

    print(f"Scanning {len(docs)} doc(s) for src/scripts anchors:")
    for d in docs:
        print(f"  {_rel(d)}")

    misses = find_dead_anchors(docs)
    if not misses:
        print("\nCLEAN — every referenced src/scripts path resolves.")
        return 0

    print(f"\nDEAD ANCHORS — {len(misses)} unresolved reference(s):\n")
    for m in misses:
        print(f"  {m['doc']}:{m['line']} -> missing {m['path']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
