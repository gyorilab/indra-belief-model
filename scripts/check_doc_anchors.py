"""Guard the live research corpus against stale code citations.

The consistency-reconciliation audit found docs hardcoding source paths and
basenames (often with a `:line` suffix) that the code had since moved or deleted —
e.g. the removed `src/indra_belief/composed_scorer.py`, still cited in the
calibration doc long after it was gone. Nothing failed CI when an anchor went
stale, so drift accumulated silently. This guard scans the live research docs,
extracts referenced Python/TypeScript/Svelte paths, asserts each explicit
repo-relative path exists, rejects numeric line anchors on either a path or a
basename, and asserts that every DOTTED symbol citation it can resolve still
resolves in the tree. It is wired into pytest via tests/test_doc_anchors.py,
mirroring how scripts/check_contamination.py is wired via
tests/test_contamination_guard_sources.py, and into CI as its own step.

WHY DOCS IS A GLOB, NOT A LIST. It used to be a hand-maintained three-element
literal. That made "unguarded" the DEFAULT for a new document: at 96cc1b7 the
list named three of seventeen live research docs, and the two newest — written the
day before, 1,515 lines between them — were read by no guard, no test and no CI
step. The doc-drift audit then found four dead symbols and 66 numeric anchors
outside the three. A glob inverts the default: a research doc is CHECKED on the
commit that adds it, and dropping a scratch .md into research/ turns CI red. That
is the correct trade.

Scope:
  * Every `research/*.md`. The glob is deliberately NON-recursive, which is what
    keeps `research/archive/**` out — completed-phase records are history and are
    never scanned.
  * Extracts explicit `src/`, `scripts/`, and `viewer/src/` paths ending in .py,
    .ts, or .svelte.
  * A referenced file that does not exist is a DEAD ANCHOR.
  * A live `file.py:123` / `path/file.ts:123-140` citation is an UNSTABLE
    ANCHOR even when the citation uses only a basename or source-path suffix;
    cite the file plus a symbol/component name instead.
  * A dotted citation whose head resolves to a repo module or class must have a
    tail that resolves too — see `find_dead_symbols`.
  * URL segments, dotted version notation, and prose such as `schema_version: 8`
    are not source anchors.

What is deliberately NOT treated as a dead anchor (neither is drift):
  * A line in an explicitly SUPERSEDED / historical context — either the line
    itself carries "(superseded)"/"(historical)"/"SUPERSEDED", or it sits under
    a Markdown heading whose text contains "superseded"/"historical" (until the
    next heading). Kept to those explicit markers so live anchors stay visible.
  * A path the docs explicitly say to BUILD — a to-be-created artifact, e.g.
    "**DO.** Build `scripts/probe_effective_context.py`". A planned file is a
    plan for future work, not a stale reference to something that used to exist.
    Build intent is scoped to the document that declares it.
  * An anchor listed in `GRANDFATHERED_ANCHORS` — pre-existing debt owned by
    somebody else, allowed at an EXACT count that can only shrink.

WHAT THIS GUARD CANNOT DO, stated here because a green run will otherwise be read
as a correct document. THIS IS A REFERENTIAL CHECK, NOT A SEMANTIC ONE.
  * Tier A — below the grammar. A BARE undotted name is invisible to the symbol
    check: `_CREDIBLE_LLM_CONF` was deleted from the tree and sat in
    serving_architecture.md §3 F8 as live code through a green run, because the
    matcher requires a dot. Widening the grammar was measured and rejected: bare
    snake_case citations are 51% unresolvable (they are third-party flags, run
    ids and JSON keys, not drift), SCREAMING_SNAKE is 11-of-12 false positives
    (shell env vars from the runbooks), and even the narrowest leading-underscore
    cut is ~50% precision. The limit is recorded rather than chased.
  * Tier B — at the grammar and semantically empty. Every path and symbol in a
    sentence can resolve while the sentence is false: a finding still headed
    `[R] HIGH` three commits after it was closed, a "Fix (small, surgical)" item
    still in the imperative after it landed. Nothing static catches this. A human
    re-reading the doc against the tree does, which is what
    `research/kernel_unification_findings.md` §7.2 is.
  * Tier C — coverage this guard admits. `symbol_coverage` reports the
    checked/unchecked split; roughly half of a typical document's dotted
    citations are unchecked (ambiguous heads, non-repo heads, chains deeper than
    `module.Class.member`). Any summary that drops that split is a misreading.
  * TypeScript and Svelte citations are path-checked and NEVER symbol-checked:
    `SYMBOL_ROOTS` is Python-only and `_tree_index` walks `rglob("*.py")`.

Exit codes:
    0  every live anchor resolves, is stable, and every checked symbol resolves
    1  an invalid anchor (missing file, or a numeric line coordinate)
    3  a `GRANDFATHERED_ANCHORS` entry is stale — the debt SHRANK (or the key is
       gone). The table may only ratchet down; paste the printed count in.
    4  a document cites a symbol that no longer resolves in the tree

4 is distinct from 1 because the repair differs in kind: a numeric anchor is fixed
by deleting a coordinate, mechanically, while a dead symbol is fixed by finding out
where the logic went and renaming the citation to its current owner. 3 is distinct
from both because nothing is wrong with the DOCUMENT — the table is out of date.

Usage:
    python scripts/check_doc_anchors.py [DOC ...]
"""
from __future__ import annotations

import ast
import re
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The live research corpus. NON-recursive on purpose: research/archive/** is
# history and is never scanned.
DOC_ROOT = ROOT / "research"
DOCS = tuple(sorted(DOC_ROOT.glob("*.md")))

# Pre-existing numeric debt, owned by whoever wrote the document, not by the
# nodes touching it now. Keys are `<cited-path>:<coord>` — never a document line
# number, because a guard that forbids citing by line must not cite by line. The
# value is the occurrence COUNT, so a repair cannot be silently traded for a new
# anchor somewhere else in the same file.
#
# THIS TABLE MAY ONLY SHRINK. An actual count above its entry reports the excess
# as a normal invalid anchor (exit 1); an actual count BELOW it, or a key with no
# occurrences left, exits 3 and prints the number to paste in. A document absent
# from this table is allowed ZERO, which is the default that covers sixteen of
# the seventeen live docs.
#
# `research/serving_architecture.md` carried twelve of these and is deliberately
# NOT here: the doc-drift audit converted all twelve to symbol citations, so it
# now joins the scan clean rather than grandfathered.
#
# `research/ben_gyori_deck_outline.md` carries the remaining 54 occurrences over
# 33 distinct keys. It is a dated deck handoff ("CURRENT DECK CONTRACT —
# 2026-07-13") for a deck that has shipped, and it is now the SOLE holder of
# anchor debt in the corpus. Retiring it removes 54 anchors by removing a
# quantity rather than by grandfathering one. That is an owner decision, not
# this guard's; until it is taken, the debt is explicit here. (research/archive/
# was itself removed, so archiving is no longer the cheap route — the choice is
# repair the 54 anchors or retire the document.)
GRANDFATHERED_ANCHORS: dict[str, dict[str, int]] = {
    "research/ben_gyori_deck_outline.md": {
        "analyze_external_gold.py:5": 1,
        "calibration_constants.py:5-6": 1,
        "calibration_constants.py:16-17": 1,
        "calibration_constants.py:41-44": 3,
        "calibration_constants.py:41-66": 7,
        "calibration_constants.py:42": 1,
        "calibration_constants.py:42-43": 1,
        "calibration_constants.py:43": 1,
        "calibration_constants.py:48": 1,
        "calibration_constants.py:48-50": 2,
        "calibration_constants.py:49": 1,
        "calibration_constants.py:50-62": 1,
        "calibration_constants.py:54": 1,
        "calibration_constants.py:55": 1,
        "calibration_constants.py:56": 1,
        "calibration_constants.py:59": 1,
        "calibration_ship_gate.py:26-32": 2,
        "calibration_ship_gate.py:67-68": 2,
        "calibration_ship_gate.py:302": 2,
        "constants.py:5-6": 1,
        "constants.py:16-17": 1,
        "noise_model.py:3-9": 1,
        "noise_model.py:5-6": 2,
        "noise_model.py:24-25": 1,
        "noise_model.py:285": 2,
        "noise_model.py:321": 2,
        "noise_model.py:354-357": 2,
        "statement_belief.py:10-12": 2,
        "statement_belief.py:51": 1,
        "statement_belief.py:74": 2,
        "statement_belief.py:75": 2,
        "statement_belief.py:145": 2,
        "statement_belief.py:221-222": 2,
    },
}

# An explicit repository-rooted code path plus an optional numeric line
# coordinate. The left boundary prevents a path segment inside a URL from being
# mistaken for a repository reference.
_ANCHOR = re.compile(
    r"(?<![A-Za-z0-9_./:-])"
    r"(?P<path>(?:(?:viewer/)?src|scripts)/[A-Za-z0-9_./+\[\]-]+\.(?:py|ts|svelte))"
    r"(?P<line>:\d+(?:-\d+)?(?![\d.]))?"
)

# A numeric source citation may use just a basename (`noise_model.py:52-76`) or
# a source-path suffix (`scorers/monolithic/scorer.py:59`). Requiring a real
# source extension and a numeric suffix avoids matching version labels, schema
# numbers, ratios, and other non-source prose. The boundary also excludes URL
# path segments such as https://example.org/src/example.py:12.
_NUMERIC_SOURCE_ANCHOR = re.compile(
    r"(?<![A-Za-z0-9_./:-])"
    r"(?P<path>(?:[A-Za-z0-9_+\[\]-]+/)*[A-Za-z0-9_+\[\]-]+\.(?:py|ts|svelte))"
    r"(?P<line>:\d+(?:-\d+)?)(?![\d.])"
)

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


def _planned_paths(records: list[tuple[str, int, str, bool]]) -> dict[str, set[str]]:
    """Per-document paths the docs explicitly say to BUILD — plans, not anchors.

    A path is 'planned' if any line in THE SAME DOCUMENT referencing it also
    carries a build intent ('Build `x`', 'new script', ...). Collected across all
    of that document's lines so a later bare reference to the same planned file
    (e.g. an '**Artifacts.**' line) is not mistaken for a dead anchor — but NOT
    across documents, so one doc's plan cannot excuse another doc's dead anchor.
    """
    planned: dict[str, set[str]] = {}
    for doc, _lineno, text, _skipped in records:
        if _BUILD_INTENT.search(text):
            for m in _ANCHOR.finditer(text):
                planned.setdefault(doc, set()).add(m.group("path"))
    return planned


def _anchor_key(path: str, coord: str) -> str:
    """`<cited-path>:<coord>` — the `GRANDFATHERED_ANCHORS` key form."""
    return f"{path}{coord}"


def find_dead_anchors(paths: list | None = None) -> list[dict]:
    """Return dead-anchor records for the given docs (defaults to DOCS).

    Each record is {"doc": str (repo-relative), "line": int, "path": str,
    "reason": str}, plus "coord" on a numeric miss. An empty list means every
    live anchor resolves, is stable, and the grandfather table is exact.
    Importable so the pytest guard can call it directly.
    """
    if paths is None:
        paths = DOCS

    records: list[tuple[str, int, str, bool]] = []  # (doc, lineno, text, skip)
    misses: list[dict] = []
    scanned: list[str] = []
    for p in paths:
        p = Path(p)
        doc_rel = _rel(p)
        if not p.exists():
            # A missing DOC is itself a drift signal — report it explicitly.
            misses.append({"doc": doc_rel, "line": 0, "path": "<doc not found>",
                           "reason": "missing document"})
            continue
        scanned.append(doc_rel)
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
        seen: set[tuple[int, str, str | None]] = set()
        for m in _ANCHOR.finditer(text):
            rel = m.group("path")
            seen.add((m.start(), rel, m.group("line")))
            if rel in planned.get(doc_rel, ()):
                continue
            if not (ROOT / rel).exists():
                misses.append({"doc": doc_rel, "line": lineno, "path": rel,
                               "reason": "missing file"})
            elif m.group("line"):
                misses.append({"doc": doc_rel, "line": lineno, "path": rel,
                               "coord": m.group("line"),
                               "reason": f"unstable numeric anchor {m.group('line')}"})
        for m in _NUMERIC_SOURCE_ANCHOR.finditer(text):
            rel = m.group("path")
            if (m.start(), rel, m.group("line")) in seen:
                continue
            misses.append({"doc": doc_rel, "line": lineno, "path": rel,
                           "coord": m.group("line"),
                           "reason": f"unstable numeric anchor {m.group('line')}"})
    return _apply_grandfather(misses, scanned)


def _apply_grandfather(misses: list[dict], scanned: list[str]) -> list[dict]:
    """Subtract the allowed pre-existing debt and enforce the shrink-only ratchet.

    Excess occurrences beyond a key's allowance are reported as ordinary invalid
    anchors. A key whose actual count fell BELOW its allowance is reported as a
    stale table entry carrying the new number, because a table that silently
    tolerates a repair lets the next regression spend the slack.
    """
    actual: dict[tuple[str, str], list[dict]] = {}
    for m in misses:
        if "coord" not in m:
            continue
        actual.setdefault((m["doc"], _anchor_key(m["path"], m["coord"])), []).append(m)

    kept: list[dict] = []
    allowed_ids: set[int] = set()
    for (doc_rel, key), rows in actual.items():
        allowance = GRANDFATHERED_ANCHORS.get(doc_rel, {}).get(key, 0)
        for row in rows[:allowance]:
            allowed_ids.add(id(row))
    for m in misses:
        if id(m) not in allowed_ids:
            kept.append(m)

    for doc_rel in scanned:
        for key, allowance in GRANDFATHERED_ANCHORS.get(doc_rel, {}).items():
            seen_count = len(actual.get((doc_rel, key), ()))
            if seen_count < allowance:
                kept.append({
                    "doc": doc_rel, "line": 0, "path": key,
                    "reason": (
                        f"stale grandfather entry — allowed {allowance}, found "
                        f"{seen_count}. The debt shrank; set it to {seen_count}"
                        + (" (delete the key)" if seen_count == 0 else "")
                        + " in GRANDFATHERED_ANCHORS"
                    ),
                })
    return kept


# ── symbol resolution ────────────────────────────────────────────────────────
#
# Moved here from scripts/check_new_section_anchors.py, whose own docstring named
# this file as where it belonged: one guard owns the whole-document mode, the
# section-scoped mode and the symbol mode instead of them being split across two
# scripts of which only one ran corpus-wide. The section-scoped script now imports
# these names rather than defining them.
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

    Fenced blocks are read as well as inline spans: the serving doc's `## 1.`
    architecture diagram is a fenced block, and four of the fourteen dead
    citations the symbol check was built for lived in it. Chains whose head is not
    a repo module or class fall out as UNCHECKED, so
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


def find_dead_symbols(doc) -> list[dict]:
    """Symbols the document cites that no longer resolve. Whole document.

    Each record is {"doc": str, "line": int, "symbol": str, "reason": str} —
    the same shape `find_dead_anchors` returns, so both can be printed by one
    loop and neither has to know about the other.
    """
    lines = Path(doc).read_text().splitlines()
    rel = _rel(doc)
    misses, seen = [], set()
    for lineno, chain in _citations(lines):
        verdict, detail = resolve_symbol(chain)
        if verdict == "missing" and (lineno, chain) not in seen:
            seen.add((lineno, chain))
            misses.append({"doc": rel, "line": lineno, "symbol": chain,
                           "reason": f"dead symbol — {detail}"})
    return misses


def symbol_coverage(doc) -> dict[str, int]:
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


def find_all_dead_symbols(paths: list | None = None) -> list[dict]:
    """`find_dead_symbols` over every doc in `paths` (defaults to DOCS)."""
    dead: list[dict] = []
    for p in (DOCS if paths is None else paths):
        if Path(p).exists():
            dead.extend(find_dead_symbols(p))
    return dead


def main(argv: list | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    docs = [Path(a) for a in argv] if argv else list(DOCS)

    print(f"Scanning {len(docs)} research doc(s) for stable code anchors and "
          f"live symbols:")
    for d in docs:
        print(f"  {_rel(d)}")

    misses = find_dead_anchors(docs)
    stale_table = [m for m in misses if m["reason"].startswith("stale grandfather")]
    anchor_misses = [m for m in misses
                     if not m["reason"].startswith("stale grandfather")]
    dead_symbols = find_all_dead_symbols(docs)
    coverage = {"checked": 0, "unchecked": 0}
    for d in docs:
        if Path(d).exists():
            c = symbol_coverage(d)
            coverage["checked"] += c["checked"]
            coverage["unchecked"] += c["unchecked"]

    if anchor_misses:
        print(f"\nINVALID ANCHORS — {len(anchor_misses)} reference(s):\n")
        for m in anchor_misses:
            print(f"  {m['doc']}:{m['line']} -> {m.get('reason', 'invalid')}: {m['path']}")
    if stale_table:
        print(f"\nSTALE GRANDFATHER TABLE — {len(stale_table)} entry(ies). The "
              f"table in {Path(__file__).name} may only shrink:\n")
        for m in stale_table:
            print(f"  {m['doc']} -> {m['reason']}: {m['path']}")
    if dead_symbols:
        print(f"\nDEAD SYMBOLS — {len(dead_symbols)} citation(s) name something the "
              f"tree no longer defines. Rewrite each to its CURRENT owner, by "
              f"symbol:\n")
        for m in dead_symbols:
            print(f"  {m['doc']}:{m['line']} -> {m['reason']}: {m['symbol']}")

    grandfathered = sum(
        sum(entries.values())
        for doc_rel, entries in GRANDFATHERED_ANCHORS.items()
        if doc_rel in {_rel(d) for d in docs}
    )
    if anchor_misses:
        return 1
    if stale_table:
        return 3
    if dead_symbols:
        return 4
    print(f"\nCLEAN — every live code anchor resolves and uses stable symbols "
          f"({grandfathered} grandfathered occurrence(s) allowed and all still "
          f"present); the symbol check read {coverage['checked']} of "
          f"{coverage['checked'] + coverage['unchecked']} dotted citations and "
          f"all resolve ({coverage['unchecked']} unchecked — see the docstring's "
          f"Tier C).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
