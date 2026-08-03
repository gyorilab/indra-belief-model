"""Section-scoped anchor guard for research/serving_architecture.md. SUPERSEDED.

WHAT CHANGED, and read this before trusting anything below it. Two things this
module was built on are no longer true:

  * The premise. `scripts/check_doc_anchors.py` could not take this document into
    its `DOCS` list because sections 1-8 carried pre-existing unstable numeric
    anchors, and repairing them belonged to the modularity audit rather than to
    the nodes appending new sections. **That debt is now ZERO** — the doc-drift
    audit converted all of it to symbol citations — so `check_doc_anchors.py`
    scans this document like any other `research/*.md`, unscoped and ungrandfathered.
  * The count. This docstring used to say the debt was "16 of them, by this
    module's own `find_dead_anchors` outside the guarded spans, so the count is
    re-derivable rather than remembered". Re-deriving it at 96cc1b7 gave **12**,
    not 16, against a byte-identical document and a byte-identical guard. The
    module whose purpose is stopping remembered-not-derived numbers carried one.
    It is recorded here rather than quietly corrected.

So this file is a compatibility surface, not the guard. `check_doc_anchors.py`
owns `DOCS` (a `research/*.md` glob), the grandfather table and the symbol check;
this module imports those and keeps only the title-scoped machinery plus the
5-code exit contract that `tests/test_doc_anchors.py` pins. Retiring it means
retargeting those tests, which is a `tests/` change and out of scope for the pass
that wrote this note. Until then: a new section appended to
research/serving_architecture.md is checked TWICE, by this module's scoped run and
by the corpus-wide run, and neither can pass what the other fails.

Historically: this guard reused `find_dead_anchors` unchanged and scoped it to
the sections the serving-architecture hypergraph appends — a new section could not
introduce anchor drift, while the older debt stayed visible and untouched.

Sections are located by TITLE, never by section number. Sibling nodes append to
this same document in the same wave, so the integer is resolved at write time and
may be renumbered later; the title is the stable identity. A sibling adds its own
section by appending one entry to `SECTION_TITLES` rather than cloning this file.

Title scoping fails OPEN on its own: a section nobody registered is not clean, it
is UNREAD, and this guard used to print OK over anchors it had never opened. So
`unregistered_sections` asserts COVERAGE — every `## ` heading from the first
owned section to end of file must be claimed by `SECTION_TITLES` — and that is
what makes the anchor result above it mean anything.

THE PATH CHECK IS NOT A SYMBOL CHECK, and that was a third way to fail open.
`find_dead_anchors` asserts that a cited FILE exists; it says nothing about the
name cited inside it. So after four assembler methods were deleted from
`comparison/replay.py`, fourteen citations of `ReplayIndex.main_request`,
`ReplayIndex._record` and `ScoringRecord.format_user_message` passed this guard at
exit 0 — every one of them named a file that still existed. `find_dead_symbols`
below closes that: a cited symbol must resolve in the tree.

    WHAT IS GUARDED — a DOTTED chain inside a backtick span or a fenced code
    block, e.g. `ReplayIndex.main_request`, `PreparedCall.client_kwargs`,
    `metrics._rankdata_avg`, whose HEAD resolves unambiguously to one repo module
    (by dotted path under src/, or by a basename unique across src/ and scripts/)
    or to one top-level class. The tail is then resolved through that module's
    top-level names — definitions AND imports, so a re-exported name still counts
    — and one level into a class body.

    WHAT IS NOT GUARDED, and no result here should be read as covering it:
      * a BARE name — `parse_response`, `_select_examples`, `price_for`. Prose is
        full of words that look like identifiers; checking them would fail on
        English, not on drift.
      * a symbol whose head is ambiguous or unknown — `scorer._select_examples`
        (two `scorer.py` in the tree), `usage.prompt_tokens` (no such module).
        Reported as UNCHECKED, never as clean.
      * a symbol named in prose without backticks.
      * anything deeper than `module.Class.member`, and any tail reached through
        an imported name — resolution stops rather than chasing into another file.
    `symbol_coverage` returns the checked/unchecked split so a caller can see how
    much of the document this actually reads.

    SCOPE differs from the anchor check ON PURPOSE. Anchors were scoped to the
    registered sections because §1-§8 carried pre-existing NUMERIC debt this guard
    deliberately left visible. There was no equivalent pre-existing SYMBOL debt in
    THIS document — the modularity audit repaired all fourteen — so the symbol
    check reads the WHOLE document. That scoping sentence was also the corpus's
    most-read statement of symbol coverage, and it was true of one file: run over
    all seventeen research docs, the same check reported FOUR dead symbols in two
    other documents, one of them a live operational runbook. Corpus-wide symbol
    checking is now `check_doc_anchors.py`'s job.

The real fix was a registry inside `check_doc_anchors.py` so one guard owns the
whole-doc, the section-scoped and the symbol modes instead of them being
duplicated here. That is done: `DOCS` is a `research/*.md` glob there, the symbol
machinery moved there, and this module imports it.

Exit codes:
    0  every checked section is clean
    1  a checked section carries an invalid anchor (each one printed)
    2  a registered section's heading is absent from the document
    3  the document carries a section no registered title claims (unchecked)
    4  the document cites a symbol that no longer resolves in the tree

4 is a distinct code rather than folded into 1 because the repair is different in
kind: a numeric anchor is fixed by deleting a coordinate, mechanically, while a
dead symbol is fixed by finding out where the logic went and renaming the citation
to its current owner. 1 still means exactly what it meant before it existed.

Usage:
    python scripts/check_new_section_anchors.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_doc_anchors import (  # noqa: E402
    SYMBOL_ROOTS,
    find_dead_anchors,
    resolve_symbol,
)
from check_doc_anchors import find_dead_symbols as _find_dead_symbols  # noqa: E402
from check_doc_anchors import symbol_coverage as _symbol_coverage  # noqa: E402

__all__ = [
    "DOC", "SECTION_TITLES", "SYMBOL_ROOTS", "find_dead_anchors",
    "find_dead_symbols", "resolve_symbol", "scoped_misses", "section_span",
    "symbol_coverage", "top_level_headings", "unregistered_sections",
]

DOC = ROOT / "research" / "serving_architecture.md"

# Titles of the appended sections this guard owns, without their numbers.
SECTION_TITLES: tuple[str, ...] = (
    "Prefix-cache benchmark specification",
    "Partition identity — what sharding would require",
)


def _heading_matcher(title: str) -> re.Pattern[str]:
    """`## <n>. <title>` — the number is matched but never used as identity."""
    return re.compile(rf"^##\s+\d+\.\s+{re.escape(title)}\s*$")


def top_level_headings(lines: list[str]) -> list[tuple[int, str]]:
    """1-based (line, text) of every `## ` heading outside a fenced code block.

    Fenced lines are skipped so a shell comment inside one of the benchmark
    command blocks can never be read as a section boundary. `### ` subsections do
    not match: the prefix is `"## "` exactly. Span-finding and the coverage
    assertion both read headings from here, so they cannot disagree about what a
    section boundary is.
    """
    headings, fenced = [], False
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced and line.startswith("## "):
            headings.append((i, line.rstrip()))
    return headings


def section_span(lines: list[str], title: str) -> tuple[int, int] | None:
    """1-based (first, last) line span of `## <n>. <title>`, or None if absent.

    The span ends at the next `## ` heading so a later appended section is not
    silently attributed to this one.
    """
    headings = top_level_headings(lines)
    match = _heading_matcher(title)
    for i, (lineno, text) in enumerate(headings):
        if match.match(text):
            nxt = headings[i + 1][0] if i + 1 < len(headings) else None
            return lineno, (nxt - 1 if nxt is not None else len(lines))
    return None


def unregistered_sections(
    doc: Path = DOC, titles: tuple[str, ...] = SECTION_TITLES
) -> list[tuple[int, str]]:
    """`## ` headings at or after the first owned section that no title claims.

    Coverage, not correctness. `scoped_misses` only ever looks INSIDE the spans
    named in `SECTION_TITLES`, so an append that forgets to register its title is
    not checked at all — and a guard that stays silent about what it did not read
    is worse than no guard. Everything from the first owned heading to end of file
    is new-section territory: claim it or the run fails.

    The window starts at the first OWNED heading rather than at the top of the
    file because §1-§8 and the References block carry the pre-existing anchor debt
    this guard deliberately leaves visible and untouched. One consequence worth
    knowing before it surprises anyone: a future section inserted ABOVE
    `## References` moves the window up and makes References itself unregistered.
    That is a real report, not a false alarm — the fix is to append below it, or to
    register it.

    Returns 1-based (line, heading text) pairs in document order; empty when no
    owned heading is present at all (`scoped_misses` reports that as absent).
    """
    headings = top_level_headings(doc.read_text().splitlines())
    matchers = [_heading_matcher(t) for t in titles]
    claimed = [any(m.match(text) for m in matchers) for _, text in headings]
    if not any(claimed):
        return []
    first_owned = claimed.index(True)
    return [
        (lineno, text)
        for (lineno, text), ok in zip(headings[first_owned:], claimed[first_owned:])
        if not ok
    ]


# ── symbol resolution — MOVED ────────────────────────────────────────────────
#
# `SYMBOL_ROOTS`, `_tree_index`, `resolve_symbol`, `find_dead_symbols` and
# `symbol_coverage` used to be defined here. They now live in
# `scripts/check_doc_anchors.py`, which runs them over the WHOLE research corpus
# rather than over this one document — the move this module's docstring asked for.
# The two wrappers below keep this module's `DOC`-defaulted signatures, which the
# pytest guard and the exit-code contract are written against.


def find_dead_symbols(doc: Path = DOC) -> list[dict]:
    """`check_doc_anchors.find_dead_symbols`, defaulted to this module's DOC."""
    return _find_dead_symbols(doc)


def symbol_coverage(doc: Path = DOC) -> dict[str, int]:
    """`check_doc_anchors.symbol_coverage`, defaulted to this module's DOC."""
    return _symbol_coverage(doc)


def _rel_doc(doc: Path) -> str:
    try:
        return str(Path(doc).resolve().relative_to(ROOT))
    except ValueError:
        return str(doc)


def scoped_misses(doc: Path = DOC, titles: tuple[str, ...] = SECTION_TITLES):
    """Return (misses inside the owned sections, titles whose heading is absent)."""
    lines = doc.read_text().splitlines()
    spans, absent = [], []
    for title in titles:
        span = section_span(lines, title)
        (absent.append(title) if span is None else spans.append(span))
    misses = [
        m for m in find_dead_anchors([doc])
        if any(lo <= m["line"] <= hi for lo, hi in spans)
    ]
    return misses, absent


def main() -> int:
    # DOC is read through the module global (not a default argument) so the
    # exit-code contract can be pinned against a synthetic document in pytest.
    misses, absent = scoped_misses(DOC)
    unregistered = unregistered_sections(DOC)
    dead_symbols = find_dead_symbols(DOC)
    coverage = symbol_coverage(DOC)

    for title in absent:
        print(f"MISSING SECTION — no heading titled {title!r} in {DOC.name}")
    for lineno, text in unregistered:
        print(f"UNREGISTERED SECTION — {DOC.name}:{lineno} {text!r} sits at or "
              f"after the first guarded section, so its anchors were NEVER "
              f"checked. Add its title to SECTION_TITLES in {Path(__file__).name} "
              f"(or place the guarded sections after it).")
    if misses:
        print(f"INVALID ANCHORS in appended sections — {len(misses)} reference(s):\n")
        for m in misses:
            print(f"  {m['doc']}:{m['line']} -> {m.get('reason', 'invalid')}: {m['path']}")
    if dead_symbols:
        print(f"DEAD SYMBOLS — {len(dead_symbols)} citation(s) name something the "
              f"tree no longer defines. Rewrite each to its CURRENT owner, by "
              f"symbol:\n")
        for m in dead_symbols:
            print(f"  {m['doc']}:{m['line']} -> {m['reason']}: {m['symbol']}")

    # Every failure is printed above; the code below names the primary repair.
    # 3 is distinct from 2 because the two are duals with opposite fixes — 2 says
    # the registry names a section the document lacks (fix the document or the
    # title), 3 says the document carries a section the registry lacks (fix the
    # registry). A rename raises both, and 2 is the more informative diagnosis.
    # 4 comes last so the codes 0-3 keep the exact meaning they had before the
    # symbol check existed.
    if absent:
        return 2
    if unregistered:
        return 3
    if misses:
        return 1
    if dead_symbols:
        return 4
    print(f"OK — appended sections of {DOC.name} carry zero invalid code anchors "
          f"({', '.join(SECTION_TITLES)}); whole-document symbol check read "
          f"{coverage['checked']} of {coverage['checked'] + coverage['unchecked']} "
          f"dotted citations and all resolve "
          f"({coverage['unchecked']} unchecked — see find_dead_symbols)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
