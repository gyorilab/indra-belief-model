"""Section-scoped anchor guard for research/serving_architecture.md.

`scripts/check_doc_anchors.py` cannot simply take this document into its `DOCS`
list: sections 1-8 carry pre-existing unstable numeric anchors — 16 of them, by
this module's own `find_dead_anchors` outside the guarded spans, so the count is
re-derivable rather than remembered — and repairing them belongs to the modularity
audit, not to the nodes appending new sections.
So this guard reuses that module's `find_dead_anchors` unchanged and scopes it to
the sections the serving-architecture hypergraph appends — a new section cannot
introduce anchor drift, while the older debt stays visible and untouched.

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

    SCOPE differs from the anchor check ON PURPOSE. Anchors are scoped to the
    registered sections because §1-§8 carry pre-existing NUMERIC debt this guard
    deliberately leaves visible. There is no equivalent pre-existing SYMBOL debt —
    the modularity audit repaired all fourteen — so the symbol check reads the
    WHOLE document. Scoping it to the tail would have left the §1 diagram and the
    §4 refactor plan, where half the dead citations were, permanently unread.

The real fix is a `SECTION_SCOPED_DOCS` registry inside `check_doc_anchors.py` so
one guard owns the whole-doc, the section-scoped and the symbol modes instead of
them being duplicated here; that file is out of scope for the nodes appending
sections, and remains the right place for this to end up.

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

import ast
import re
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_doc_anchors import find_dead_anchors  # noqa: E402

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


# ── symbol resolution ────────────────────────────────────────────────────────
#
# The tree is read through `ast` rather than imported: importing
# `indra_belief.scorers.monolithic.scorer` to ask whether it defines a name would
# resolve MONO_VARIANT from the ambient environment and pull in the heavy
# closure. A guard must not have opinions that depend on how it was invoked.

SYMBOL_ROOTS = ("src", "scripts")

# A file extension in the tail position means the chain is a FILENAME, not a
# symbol — `cost.py`, `manifest.json`. Without this, `cost.py` resolves its head
# to src/indra_belief/corpus/cost.py and then reports the absence of a name `py`.
_FILE_SUFFIXES = frozenset({
    "py", "json", "jsonl", "md", "ts", "svelte", "txt", "sh", "yml", "yaml",
    "toml", "csv", "html", "ndjson", "lock", "cfg", "ini", "log",
})

# A dotted chain of Python identifiers. The left boundary rejects a chain that is
# really a path segment (`corpus/cost.py`) or the tail of a longer word.
_DOTTED = re.compile(
    r"(?<![A-Za-z0-9_./])"
    r"(?P<chain>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)"
)
_BACKTICK_SPAN = re.compile(r"`([^`]+)`")


@lru_cache(maxsize=1)
def _tree_index() -> tuple[dict[str, Path], dict[str, tuple[Path, ...]], frozenset[str]]:
    """(module name -> file, class name -> defining files, package names).

    Modules are indexed under their dotted path relative to a root (so
    `indra_belief.comparison.replay`) and under every SUFFIX of it that is unique
    across the whole scan — `comparison.replay`, `replay` — because that is how
    this repository's prose actually cites them. An ambiguous suffix is
    deliberately absent: `scorer` names two files, so a citation using it is
    reported UNCHECKED rather than resolved against a coin flip.
    """
    modules: dict[str, Path] = {}
    suffixes: dict[str, list[Path]] = {}
    classes: dict[str, list[Path]] = {}
    packages: set[str] = set()
    for root_name in SYMBOL_ROOTS:
        root = ROOT / root_name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(root).with_suffix("")
            dotted = ".".join(rel.parts)
            if dotted.endswith(".__init__"):
                packages.add(dotted[: -len(".__init__")])
                dotted = dotted[: -len(".__init__")]
            modules.setdefault(dotted, path)
            parts = dotted.split(".")
            for start in range(1, len(parts)):
                suffixes.setdefault(".".join(parts[start:]), []).append(path)
            for part_count in range(1, len(rel.parts)):
                packages.add(".".join(rel.parts[:part_count]))
            try:
                tree = ast.parse(path.read_text())
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    classes.setdefault(node.name, []).append(path)
    # Package suffixes too, so `scorers.monolithic` resolves the way it is cited.
    package_suffixes: dict[str, list[str]] = {}
    for package in packages:
        parts = package.split(".")
        for start in range(1, len(parts)):
            package_suffixes.setdefault(".".join(parts[start:]), []).append(package)
    packages |= {name for name, owners in package_suffixes.items() if len(owners) == 1}
    for name, paths in suffixes.items():
        if len(paths) == 1:
            modules.setdefault(name, paths[0])
    return (modules,
            {name: tuple(paths) for name, paths in classes.items()},
            frozenset(packages))


def _module_names(path: Path) -> tuple[frozenset[str], dict[str, ast.ClassDef]]:
    """(every name the module's top level binds, its top-level classes).

    Imports count. A module that re-exports a name — `comparison.replay` imports
    `parse_response` from `indra_belief.verdict` — genuinely provides it, and a
    doc citing `replay.parse_response` is citing something a reader can reach.
    """
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError, OSError):
        return frozenset(), {}
    names: set[str] = set()
    class_defs: dict[str, ast.ClassDef] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
            class_defs[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return frozenset(names), class_defs


def _class_members(node: ast.ClassDef) -> frozenset[str]:
    names: set[str] = set()
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(child.name)
        elif isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            names.add(child.target.id)
    return frozenset(names)


def resolve_symbol(chain: str) -> tuple[str, str]:
    """Resolve a dotted citation. Returns (verdict, detail).

    verdict is "ok", "missing", or "unchecked". "unchecked" is not a pass — it
    means this guard cannot say, and `symbol_coverage` counts it so the guard
    never claims coverage it does not have.
    """
    segments = chain.split(".")
    if segments[-1].lower() in _FILE_SUFFIXES:
        return "unchecked", "reads as a filename, not a symbol"
    modules, classes, packages = _tree_index()

    if chain in modules or chain in packages:
        return "ok", "module"

    # Longest module prefix first: `indra_belief.comparison.replay.ReplayIndex`
    # must bind to the module, not to the `indra_belief` package.
    for cut in range(len(segments) - 1, 0, -1):
        head = ".".join(segments[:cut])
        if head in modules:
            return _resolve_in_module(modules[head], segments[cut:], head)
        if head in packages:
            return "unchecked", f"{head} is a package; {segments[cut]} is not a module in it"

    head = segments[0]
    if head in classes:
        paths = classes[head]
        if len(paths) != 1:
            return "unchecked", f"class {head} is defined in {len(paths)} files"
        return _resolve_in_module(paths[0], segments, f"<{paths[0].name}>")
    return "unchecked", f"{head} is not a module or class in {'/'.join(SYMBOL_ROOTS)}"


def _resolve_in_module(path: Path, segments: list[str], where: str) -> tuple[str, str]:
    names, class_defs = _module_names(path)
    rel = path.relative_to(ROOT).as_posix()
    first = segments[0]
    if first not in names:
        return "missing", f"{first} is not defined in {rel}"
    if len(segments) == 1:
        return "ok", rel
    if first not in class_defs:
        return "unchecked", f"{first} in {rel} is imported or not a class"
    members = _class_members(class_defs[first])
    if segments[1] not in members:
        return "missing", f"{first}.{segments[1]} is not defined in {rel}"
    if len(segments) > 2:
        return "unchecked", f"deeper than {first}.{segments[1]} in {rel}"
    return "ok", rel


def _citations(lines: list[str]) -> list[tuple[int, str]]:
    """1-based (line, dotted chain) for every backticked or fenced citation.

    Fenced blocks are read as well as inline spans: the `## 1.` architecture
    diagram is a fenced block, and four of the fourteen dead citations lived in
    it. Chains whose head is not a repo module or class fall out as UNCHECKED, so
    `python -m vllm.entrypoints.openai.api_server` in a launch command costs
    nothing.
    """
    found, fenced = [], False
    for lineno, line in enumerate(lines, 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        spans = [line] if fenced else [m.group(1) for m in _BACKTICK_SPAN.finditer(line)]
        for span in spans:
            for m in _DOTTED.finditer(span):
                found.append((lineno, m.group("chain")))
    return found


def find_dead_symbols(doc: Path = DOC) -> list[dict]:
    """Symbols the document cites that no longer resolve. Whole document.

    Each record is {"doc": str, "line": int, "symbol": str, "reason": str} —
    the same shape `find_dead_anchors` returns, so both can be printed by one
    loop and neither has to know about the other.
    """
    lines = Path(doc).read_text().splitlines()
    rel = _rel_doc(doc)
    misses, seen = [], set()
    for lineno, chain in _citations(lines):
        verdict, detail = resolve_symbol(chain)
        if verdict == "missing" and (lineno, chain) not in seen:
            seen.add((lineno, chain))
            misses.append({"doc": rel, "line": lineno, "symbol": chain,
                           "reason": f"dead symbol — {detail}"})
    return misses


def symbol_coverage(doc: Path = DOC) -> dict[str, int]:
    """How many cited chains this guard actually read. Honesty, not correctness.

    A guard that reports "OK" without saying how much it declined to check
    invites exactly the misreading that let 14 dead citations ship green.
    """
    counts = {"checked": 0, "unchecked": 0, "missing": 0}
    for _lineno, chain in _citations(Path(doc).read_text().splitlines()):
        verdict, _detail = resolve_symbol(chain)
        counts["unchecked" if verdict == "unchecked" else "checked"] += 1
        if verdict == "missing":
            counts["missing"] += 1
    return counts


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
