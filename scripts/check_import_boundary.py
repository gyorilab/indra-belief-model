"""Guard the deployable core against imports from the research harness.

The repository holds two things that have to stay separable: a CORE that serves a
belief for one (Statement, Evidence) pair, and a RESEARCH harness that measures
whether that belief is any good. Research INHERITS the core — `comparison` reads
`prepared_execution`, `verdict`, `scorers`, `spend_guard`, `hashing`,
`model_client`; `results` reads `calibration_constants`, `noise_model`,
`statement_belief`. Those forward edges are correct and this guard leaves them
alone. The reverse edge is the defect: a core that reaches into the harness
cannot be deployed without it.

At the commit that added this file the tree had exactly ONE such edge —
`prepared_execution` importing `ContractError` from `comparison.contracts`, the
serving kernel reaching into the research harness for the base class of its own
`ReplayError`, in a module whose Home note already argued the ownership rule it
was breaking. `ContractError` now lives in `prepared_execution` and
`comparison.contracts` re-exports it, so no importer changed. Nothing enforced
that edge's absence, which is why it existed; this guard is the enforcement. It
is wired into pytest via tests/test_import_boundary.py, mirroring how
scripts/check_contamination.py is wired via
tests/test_contamination_guard_sources.py. CI reaches it only through that test
(`.github/workflows/ci.yml:31`, `python -m pytest -q`); unlike the contamination
and doc-anchor guards it has no step of its own in that workflow.

WHY CORE IS DERIVED AND ONLY RESEARCH IS DECLARED. `CORE` is not a list. It is
computed two ways that must agree, and both of them make "checked" the DEFAULT:

  * `core_closure()` is the transitive closure from the SERVING ENTRY POINTS —
    what `import indra_belief` and its siblings actually cost at runtime. Most of
    the entry set is read out of `src/indra_belief/__init__.py` itself (the
    module that calls its own contents "Public API"), so it tracks the package
    rather than a copy of it; `DECLARED_ENTRY_POINTS` adds the four serving
    surfaces the package `__init__` does not re-export, each with its reason.
  * `core_modules()` is every first-party module NOT under a declared research
    root. A closure alone would leave the unreached unguarded — `belief_scorer`,
    `scorers.kg_signal` and `scorers.probes.bind_check` are scorer code that no
    closure in this file reaches today — and "unguarded by default" is exactly
    how the doc guard's hand-maintained DOCS list rotted. So the guard checks the
    complement instead: a NEW module under `src/indra_belief/` is checked on the
    commit that adds it, and to be exempt it has to be declared research out loud.
    Those three modules are CHECKED, and each carries an entry in
    `UNREACHED_DISPOSITIONS` below saying what it is and why it stays. That table
    is ANNOTATION ONLY: `core_modules()` never reads it, so writing a disposition
    cannot remove a module from checking. Unreached and unchecked are different
    states, and nothing here quiets the report by declaring a module non-core.

The closure is still computed, reported, and used: a violation inside it is
marked `serving-reachable`, because an edge on the path `import indra_belief`
takes is worse than one in a module nothing loads. A SECOND closure,
`tool_closure()`, is rooted at `scripts/` and reported BESIDE it, never merged
into it: "reached by a build/analysis script" and "reached by a serving entry
point" are different claims, and merging them would launder a tool dependency
into a serving one and blunt the `serving-reachable` severity label that
`find_boundary_violations` assigns. `unreached_modules()` splits the core three
ways over those two closures and the report prints the second and third buckets
BY NAME, each script-reached module beside the script that reaches it.

THE THIRD BUCKET IS NOT A KILL LIST. It is exactly the set of core modules that
no closure IN THIS FILE reaches — one measure, run over two declared root sets.
That is a QUESTION (why does this module exist with no importer?), not a verdict,
and every member has a written answer in `UNREACHED_DISPOSITIONS` below which the
report prints beside its name. An import graph can establish that nothing here
imports a module; it cannot establish that nothing needs one. The report also
annotates each member with the TEST files that import it, derived by the same AST
over `TESTS_ENTRY_DIR` — an annotation, never a closure root, so it changes what
the report SAYS about a module and never which bucket the module is in.

WHAT THIS GUARD CANNOT SEE, stated here because a green run will otherwise be
read as "the core is deployable". THIS IS AN IMPORT-GRAPH CHECK, NOTHING MORE.

  * Paths taken as PARAMETERS.
    `src/indra_belief/calibration_constants.py::_call_log_fingerprints` opens
    whatever run file it is handed, and
    `src/indra_belief/calibration_constants.py::reader_configuration_for_run`
    reads the corresponding metadata; no literal at either call site says which
    artifact that is. That is one form of real path coupling, and this
    single-file AST cannot recover where the argument came from. What syntax CAN
    show is measured instead: `data_path_literals()` reports strings shaped
    like a repo-relative `data/` path, split by recognised call position.
    Today all ten live in `calibration_constants`'s provenance dicts
    (`_PROFILE_META`, `_named_profile`) and none sits in a recognised call.
    Zero literals in a call position is therefore NOT zero path coupling.
    `src/indra_belief/data/corpus.py::DEFAULT_CORPUS` is the stronger assembled
    counter-example: it resolves to repo `data/` and its module is in
    `core_closure()`. `src/indra_belief/scorers/kg_signal.py::_DEFAULT_CORPUS`
    also resolves to repo `data/`, although that module is not serving-reachable.
    In contrast,
    `src/indra_belief/scorers/monolithic/scorer.py::_EXAMPLE_BANK_PATH` resolves
    to PACKAGE data. `assembled_data_paths()` derives and distinguishes those
    bases rather than treating every `/ "data"` expression alike. Prose
    `data/` mentions in comments and docstrings are not string literals and are
    correctly scored zero.
  * Dynamic imports. A module name assembled at runtime is invisible.
    `dynamic_import_sites()` reports every `importlib.import_module(...)` and
    `__import__(...)` whose argument is not a string literal, so the blind spots
    are enumerated rather than assumed empty.
  * Third-party weight. This says nothing about whether the core is SMALL, only
    about which side of the line it imports from. The size claim belongs to the
    Dockerfile's `test` stage, which builds without gilda or indra installed.
  * The entry points, the research roots and the tool root are JUDGMENTS. Four
    entry points, six research roots and one directory are declared here, each
    with the reason it is what it is. `main` fails if any of them stops naming
    something real (exit 3) but nothing can tell you they name the right ones.
  * Deleting an import is not the same as deleting a dependency: a core module
    can still reach research behavior through a callback, a registry, or a
    string, and the graph will look clean.

Function-level imports are NOT a blind spot: `import_edges` walks function
bodies, so `scorers.monolithic.scorer`'s deferred `tools.gilda_tools` import is
an edge like any other, and `coverage()` reports the module/function split.

Exit codes:
    0  no core module imports a research module
    1  a core -> research import (the boundary violation)
    3  a declaration has rotted — an entry point or research root that resolves
       to no module, a tool or tests root under which no file imports the
       package, or an `UNREACHED_DISPOSITIONS` key naming a module that is
       absent or no longer unreached. The guard will not report coverage against
       a declaration that no longer describes the tree

1 and 3 are distinct because the repair differs in kind: a violation is fixed by
moving code or inverting a dependency, while a stale declaration is fixed by
editing the table below or restoring what it points at.

There is deliberately NO exit code for the opposite failure — an unreached module
with no disposition written for it. That is a MISSING declaration, not a rotted
one: the tree is fine and the prose is behind it. `undisposed_modules()` names
them and tests/test_import_boundary.py fails on a non-empty result, which puts
the repair where the writing happens instead of stretching 3 to mean two things.

Usage:
    python scripts/check_import_boundary.py
"""
from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGE = "indra_belief"

# The package module whose docstring calls its own contents "Public API". Every
# first-party module it can bind — module level or inside `__getattr__` — is a
# serving entry point, read from the file rather than copied into a list here.
PUBLIC_API_MODULE = f"{PACKAGE}.__init__"

# Serving surfaces the package `__init__` does not re-export. Each is here
# because something in the tree treats it as an entry point, not because it
# looked core.
DECLARED_ENTRY_POINTS: Mapping[str, str] = {
    f"{PACKAGE}.scorers.scorer":
        "its own docstring: 'Evidence quality scorer entry point — two "
        "architectures available'. Dispatches monolithic and decomposed, and is "
        "how the decomposed arch the package docstring points at is reached.",
    f"{PACKAGE}.statement_belief":
        "per-evidence verdicts -> one calibrated statement belief. Serving a "
        "belief for a Statement IS this call; nothing in the package __init__ "
        "exposes it yet.",
    f"{PACKAGE}.calibration_constants":
        "produces the reader profile `statement_belief(soft=...)` consumes "
        "(`calibration_for_run`), so a served belief is calibrated by it. Fitted "
        "by research, read by the core.",
    f"{PACKAGE}.spend_guard":
        "a deployed service that dispatches to a paid provider cannot enforce a "
        "cap without it, which is also why its `corpus.cost` import is a core "
        "edge and not a violation.",
}

# The measurement side. A module under one of these prefixes exists to say
# whether a belief is good, not to produce one, and the core may not import it.
# Keep this short: everything NOT under a root here is checked, so a prefix added
# here removes guard coverage and has to earn it.
RESEARCH_ROOTS: Mapping[str, str] = {
    f"{PACKAGE}.comparison":
        "the benchmark harness — run plans, spend ledgers, frozen replay, "
        "provider dispatch for a measured arm, the comparison CLI. The most "
        "worked code in the repo and still research: it measures.",
    f"{PACKAGE}.results":
        "turns a scoring run into the viewer's per-evidence / per-statement "
        "export plus the bucket taxonomy. A reading of a run, not a run.",
    f"{PACKAGE}.metrics":
        "confusion / precision / recall / F1 / ECE over a scored eval set.",
    f"{PACKAGE}.curation":
        "INDRA curations as gold — the label side of every evaluation.",
    f"{PACKAGE}.sampling":
        "two-stage cluster sampling and interval sizing for eval queues.",
    f"{PACKAGE}.model_meta":
        "curated model parameter counts baked into a run's export for the "
        "viewer's size axis.",
}

# The build/analysis surface: every file under here is a root of `tool_closure()`
# and none of them is a serving path. The directory is the whole declaration —
# the scripts are found by glob and their imports by AST, so a script added
# tomorrow reaches what it reaches without an edit here.
#
# ROOTS ARE `scripts/` ONLY, NEVER `tests/`. Measured: tests/test_belief_scorer.py
# and tests/test_bind_check.py import two of the three modules that no closure
# reaches, so rooting a closure at tests/ would zero the orphan signal outright —
# every module a test touches would report as reached, which is the opposite of
# what "reached by nothing" is asked to mean.
TOOL_ENTRY_DIR = ROOT / "scripts"

# The test surface, and it is an ANNOTATION ROOT — NEVER A CLOSURE ROOT. Files
# under here are parsed by the same AST so the report can say WHICH test imports
# an unreached module (`test_reachers()`), and that is the whole of what this
# directory is allowed to do. It may not move a module out of the unreached
# bucket: the rule stated above at TOOL_ENTRY_DIR holds unchanged — rooting a
# closure at tests/ would zero the signal outright, because every module a test
# touches would report as reached. "A test imports it" and "a serving entry point
# or a build script reaches it" are different claims, and only the second one is
# closure. Membership of the unreached bucket is computed without this directory.
TESTS_ENTRY_DIR = ROOT / "tests"

# What each unreached module IS, and why it stays. One entry per member of the
# third bucket — the modules no closure in this file reaches.
#
# ANNOTATION ONLY. This table is read by the report and by its own stale check,
# and by NOTHING else: not `core_modules()`, not `is_research()`, not
# `core_closure()`, not `find_boundary_violations()`. Writing an entry here does
# not remove a module from checking and is not a substitute for deleting one —
# it is the answer to the question the third bucket asks.
#
# Each entry cites a file and a SYMBOL rather than a file:line, because a line
# number recorded in this file's runtime output is the thing that rots first.
UNREACHED_DISPOSITIONS: Mapping[str, str] = {
    f"{PACKAGE}.belief_scorer":
        "LIBRARY SURFACE: an `indra.belief.BeliefScorer` implementation, so its "
        "callers are OUTSIDE this repository — `BeliefEngine(scorer="
        "LLMBeliefScorer(client))` — and having no internal importer is the "
        "EXPECTED shape for a plug-in socket rather than a sign of disuse. An "
        "internal importer would mean we had started calling our own socket, "
        "which is not what implementing someone else's interface is for.",
    f"{PACKAGE}.scorers.kg_signal":
        "RESEARCH LINEAGE, UNWIRED: the U-phase KG-as-confidence-modifier, whose "
        "Q-phase verdict-override predecessor regressed -2.65pp and whose "
        "surviving contract is deliberately asymmetric — KG presence boosts "
        "confidence, KG absence is silent. Nothing calls `get_signal` today, and "
        "it is kept rather than deleted because two live things still describe "
        "it: README.md's source-tree map lists `kg_signal.py`, and "
        "`EvidenceContext.kg_signal` in scorers/context.py carries both the field "
        "and the full 16-line contract for this module's output, which would be "
        "left describing nothing.",
    f"{PACKAGE}.scorers.probes.bind_check":
        "TESTED SCAFFOLDING: Stage 2 of the CC-phase extract-then-bind-check "
        "redesign — the deterministic half deciding whether an extracted relation "
        "tuple matches the claim, with no LLM — which LOST to the shipped AA "
        "probe on holdout_cc and was kept on purpose as the measured alternative. "
        "The axis taxonomy and sign reconciliation it encodes are verified "
        "independently of the arm that lost.",
}


@dataclass(frozen=True)
class Edge:
    """One first-party import, as written."""

    importer: str
    target: str
    line: int
    scope: str  # "module" or "function"


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(SRC).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


@lru_cache(maxsize=1)
def module_table() -> Mapping[str, Path]:
    """Every first-party module name -> its file.

    A glob, deliberately: a module added under `src/indra_belief/` joins the scan
    on the commit that adds it.
    """
    table: dict[str, Path] = {}
    for path in sorted((SRC / PACKAGE).rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        table[_module_name(path)] = path
    return table


def _resolve(target: str, modules: Mapping[str, Path]) -> str | None:
    """Longest prefix of a dotted import target that is a real module.

    `indra_belief.comparison.contracts.ContractError` resolves to the module, not
    to the package, so an edge is attributed to the file it actually reaches.
    """
    parts = target.split(".")
    for cut in range(len(parts), 0, -1):
        candidate = ".".join(parts[:cut])
        if candidate in modules:
            return candidate
    return None


def import_edges(name: str, path: Path,
                 modules: Mapping[str, Path] | None = None) -> tuple[Edge, ...]:
    """First-party imports written in one module, module-level and deferred.

    Function bodies are walked. A deferred import is still a dependency — it just
    fails later — so `scope` records where it was written instead of excusing it.
    Relative imports are resolved against the module's package.
    """
    modules = module_table() if modules is None else modules
    package = name if path.name == "__init__.py" else name.rsplit(".", 1)[0]
    found: dict[tuple[str, int], Edge] = {}

    def record(target: str, line: int, depth: int) -> None:
        if not target.startswith(PACKAGE):
            return
        resolved = _resolve(target, modules)
        if resolved is None or resolved == name:
            return
        scope = "function" if depth else "module"
        key = (resolved, line)
        # One `from x import a, b` line is one edge, not one per alias.
        found.setdefault(key, Edge(name, resolved, line, scope))

    class Walk(ast.NodeVisitor):
        def __init__(self) -> None:
            self.depth = 0

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1

        visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                record(alias.name, node.lineno, self.depth)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if node.level:
                parts = package.split(".")
                base = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
                head = ".".join(base + ([node.module] if node.module else []))
            else:
                head = node.module or ""
            for alias in node.names:
                record(f"{head}.{alias.name}", node.lineno, self.depth)
            record(head, node.lineno, self.depth)

    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError, OSError):
        return ()
    Walk().visit(tree)
    return tuple(sorted(found.values(), key=lambda e: (e.line, e.target)))


@lru_cache(maxsize=1)
def import_graph() -> Mapping[str, tuple[Edge, ...]]:
    modules = module_table()
    return {name: import_edges(name, path, modules) for name, path in modules.items()}


def is_research(name: str) -> bool:
    return any(name == root or name.startswith(root + ".") for root in RESEARCH_ROOTS)


def research_modules() -> frozenset[str]:
    return frozenset(name for name in module_table() if is_research(name))


def core_modules() -> frozenset[str]:
    """Everything first-party that is not declared research. Checked by default."""
    return frozenset(name for name in module_table() if not is_research(name))


def serving_entry_points() -> tuple[str, ...]:
    """The serving surface: the package's own Public API plus the declared four.

    The first half is READ from `src/indra_belief/__init__.py` — its module-level
    imports and the ones inside `__getattr__` — so adding a name to the package's
    public API extends the closure without editing this file.
    """
    modules = module_table()
    init_path = modules.get(PACKAGE)
    derived: set[str] = set()
    if init_path is not None:
        derived = {edge.target for edge in import_edges(PACKAGE, init_path, modules)}
    return tuple(sorted(derived | set(DECLARED_ENTRY_POINTS) | {PACKAGE}))


def _with_parents(name: str, modules: Mapping[str, Path]) -> Iterable[str]:
    """A module plus the packages Python executes on the way to it."""
    parts = name.split(".")
    for cut in range(1, len(parts) + 1):
        candidate = ".".join(parts[:cut])
        if candidate in modules:
            yield candidate


def _reach(seeds: Iterable[str]) -> frozenset[str]:
    """Transitive import closure over an already-seeded worklist.

    The one traversal both closures use, so `core_closure()` and `tool_closure()`
    cannot drift into meaning different things by different code.
    """
    modules = module_table()
    graph = import_graph()
    seen: set[str] = set()
    stack = list(seeds)
    while stack:
        name = stack.pop()
        if name in seen or name not in modules:
            continue
        seen.add(name)
        for edge in graph.get(name, ()):  # noqa: SIM118 - Mapping, not dict
            stack.extend(_with_parents(edge.target, modules))
    return frozenset(seen)


@lru_cache(maxsize=1)
def core_closure() -> frozenset[str]:
    """Transitive closure from the serving entry points, parent packages included.

    Research modules are recorded as violations by `find_boundary_violations` but
    ARE traversed here, so the closure reports the honest runtime cost of an
    `import indra_belief` while the edge is still broken.

    Rooted at the SERVING entry points and nothing else. Script roots live in
    `tool_closure()`; widening this one would tell `find_boundary_violations` that
    a module a build script imports is on the path `import indra_belief` takes.
    """
    return _reach(serving_entry_points())


def _entry_points(directory: Path) -> Mapping[str, tuple[Edge, ...]]:
    """Every file under `directory` -> the first-party modules it imports.

    Each file is parsed by the same `import_edges` the package uses, under a
    synthetic `__script__.<stem>` name: scripts and tests are not importable
    modules, and the synthetic package keeps relative-import resolution and the
    self-edge check well defined without inventing a second parser.

    One reader for both declared directories. `tool_entry_points()` roots a
    CLOSURE at what this returns for `TOOL_ENTRY_DIR`; `test_reachers()` only
    ANNOTATES with what it returns for `TESTS_ENTRY_DIR` — the difference is in
    the caller, never in how the files are read.
    """
    modules = module_table()
    table: dict[str, tuple[Edge, ...]] = {}
    if not directory.is_dir():
        return table
    for path in sorted(directory.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        key = str(path.relative_to(directory))
        table[key] = import_edges(f"__script__.{path.stem}", path, modules)
    return table


@lru_cache(maxsize=1)
def tool_entry_points() -> Mapping[str, tuple[Edge, ...]]:
    """Every file under `TOOL_ENTRY_DIR` -> the first-party modules it imports."""
    return _entry_points(TOOL_ENTRY_DIR)


@lru_cache(maxsize=1)
def tool_closure() -> frozenset[str]:
    """Transitive closure from the build/analysis scripts, parent packages included.

    Reported beside `core_closure()` and never folded into it — see the module
    docstring. A module only in here is kept alive by a tool, not by a served call.
    """
    modules = module_table()
    return _reach(
        name
        for edges in tool_entry_points().values()
        for edge in edges
        for name in _with_parents(edge.target, modules)
    )


def unreached_modules() -> dict:
    """Core modules by what reaches them: serving, tool-only, neither.

    `tool_only` maps a module to the script basenames whose imports reach it, so
    the reason is read off the tree rather than off a table someone maintains.
    `unreached` is the residue — reached by no serving entry point and no script.
    That bucket is named for the MEASURE and not for a conclusion: what it holds
    is answered per module by `UNREACHED_DISPOSITIONS`, and `TESTS_ENTRY_DIR` is
    deliberately not consulted here, so a test importing a module cannot move it.
    """
    core = core_modules()
    serving = core & core_closure()
    rest = core - serving
    modules = module_table()
    tool_only: dict[str, tuple[str, ...]] = {}
    for script, edges in sorted(tool_entry_points().items()):
        if not edges:
            continue
        reached = _reach(
            name for edge in edges for name in _with_parents(edge.target, modules)
        )
        for name in sorted(rest & reached):
            tool_only[name] = tool_only.get(name, ()) + (script,)
    return {
        "serving": tuple(sorted(serving)),
        "tool_only": tool_only,
        "unreached": tuple(sorted(rest - set(tool_only))),
    }


@lru_cache(maxsize=1)
def test_reachers() -> Mapping[str, tuple[str, ...]]:
    """Unreached module -> the test files that import it, derived by AST.

    An ANNOTATION, computed only over the current unreached bucket and never fed
    back into it. The point is that the report can say "tests/test_bind_check.py
    reaches it" as a fact about the tree: rename that file and the printed line
    changes, which a sentence written into a docstring would not.

    Same parser, same `_reach`, same shape as `tool_only`, so the two lines in the
    report are the same kind of claim about different roots.
    """
    unreached = set(unreached_modules()["unreached"])
    if not unreached:
        return {}
    modules = module_table()
    reachers: dict[str, tuple[str, ...]] = {}
    for test_file, edges in sorted(_entry_points(TESTS_ENTRY_DIR).items()):
        if not edges:
            continue
        reached = _reach(
            name for edge in edges for name in _with_parents(edge.target, modules)
        )
        for name in sorted(unreached & reached):
            reachers[name] = reachers.get(name, ()) + (test_file,)
    return reachers


def undisposed_modules() -> tuple[str, ...]:
    """Unreached modules with no entry in `UNREACHED_DISPOSITIONS`.

    The one function the report and the pytest gate both read, so "the report
    printed a bare name" and "the test failed" are the same condition and cannot
    drift apart. Non-empty means the tree grew a module nothing reaches and nobody
    wrote down what it is — a missing declaration, repaired by writing one, not by
    deleting the module and not by a new exit code.
    """
    return tuple(
        name for name in unreached_modules()["unreached"]
        if name not in UNREACHED_DISPOSITIONS
    )


def _first_sentence(text: str) -> str:
    """Up to the first sentence break, for the one-line report form.

    Split on ". " and not on "." alone: the dispositions cite `README.md`,
    `indra.belief.BeliefScorer` and `-2.65pp`, none of which ends a sentence.
    """
    head = text.split(". ", 1)[0].rstrip(".")
    return f"{head}."


def find_boundary_violations(
    graph: Mapping[str, tuple[Edge, ...]] | None = None,
) -> list[dict]:
    """Every import from a core module into a research module.

    Each record is {"importer", "target", "line", "scope", "serving_reachable",
    "root", "file"}. An empty list means the core imports nothing from the
    research harness. Importable so the pytest guard can call it directly.
    """
    graph = import_graph() if graph is None else graph
    modules = module_table()
    reachable = core_closure()
    violations: list[dict] = []
    for name in sorted(graph):
        if is_research(name):
            continue
        for edge in graph[name]:
            if not is_research(edge.target):
                continue
            root = next(
                r for r in RESEARCH_ROOTS
                if edge.target == r or edge.target.startswith(r + ".")
            )
            violations.append({
                "importer": edge.importer,
                "target": edge.target,
                "line": edge.line,
                "scope": edge.scope,
                "serving_reachable": edge.importer in reachable,
                "root": root,
                "file": str(modules[name].relative_to(ROOT)),
            })
    # Serving-reachable first: an edge on the path `import indra_belief` takes is
    # the one that stops the core from deploying.
    violations.sort(key=lambda v: (not v["serving_reachable"], v["importer"], v["target"]))
    return violations


def stale_declarations() -> list[dict]:
    """Declarations that no longer describe the tree or its reachability.

    A declaration that names nothing is not coverage. That holds for the derived
    tool root too: if `TOOL_ENTRY_DIR` moves or empties, every script-reached
    module silently joins the unreached bucket and the report starts lying quietly
    rather than failing loudly, so a glob yielding no first-party importer is
    stale. `TESTS_ENTRY_DIR` takes the identical rule for the identical reason —
    it annotates rather than reaching, so an emptied tests root does not move a
    module anywhere, it just silently drops the sentence naming what covers it,
    which is the same quiet lie. A disposition key is the same rot in either
    direction: it is stale when its module was deleted or renamed, and also when
    the module is still present but has left the unreached bucket. In the latter
    case it keeps explaining why nothing reaches a module that something now
    reaches.
    """
    modules = module_table()
    stale: list[dict] = []
    for name in DECLARED_ENTRY_POINTS:
        if name not in modules:
            stale.append({"kind": "entry point", "name": name})
    for root in RESEARCH_ROOTS:
        if root not in modules and not any(
            m.startswith(root + ".") for m in modules
        ):
            stale.append({"kind": "research root", "name": root})
    unreached = set(unreached_modules()["unreached"])
    for name in UNREACHED_DISPOSITIONS:
        if name not in modules or name not in unreached:
            stale.append({"kind": "unreached disposition", "name": name})
    if not any(edges for edges in tool_entry_points().values()):
        stale.append({
            "kind": "tool root",
            "name": f"{TOOL_ENTRY_DIR.name}/ (no file under it imports {PACKAGE})",
        })
    if not any(edges for edges in _entry_points(TESTS_ENTRY_DIR).values()):
        stale.append({
            "kind": "tests root",
            "name": f"{TESTS_ENTRY_DIR.name}/ (no file under it imports {PACKAGE})",
        })
    return stale


def dynamic_import_sites() -> list[dict]:
    """`importlib.import_module` / `__import__` calls the graph cannot follow.

    A literal argument is resolvable and not reported; anything else is a hole,
    named so it is counted rather than assumed absent.
    """
    holes: list[dict] = []
    for name, path in sorted(module_table().items()):
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            called = (
                func.id if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute)
                else ""
            )
            if called not in {"import_module", "__import__"}:
                continue
            first = node.args[0] if node.args else None
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                continue
            holes.append({"module": name, "line": node.lineno, "call": called})
    return holes


def _is_data_path(node: ast.AST) -> bool:
    """A string literal shaped like a repo-relative `data/` path."""
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("data/")
    )


def _single_bindings(tree: ast.AST) -> dict[str, ast.AST]:
    """Names with exactly one direct assignment in this one syntax tree.

    Deliberately whole-file and scope-agnostic: the same spelling assigned in two
    functions is ambiguous rather than guessed. Store/delete contexts and other
    binders invalidate a candidate, so duplicate assignment, `+=`, a parameter,
    an import, or any other rebinding cannot be resolved accidentally.
    """
    counts: dict[str, int] = {}
    candidates: dict[str, ast.AST] = {}

    def record(name: str | None) -> None:
        if name:
            counts[name] = counts.get(name, 0) + 1

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            record(node.id)
        elif isinstance(node, ast.arg):
            record(node.arg)
        elif isinstance(node, ast.alias):
            bound = node.asname or node.name.split(".", 1)[0]
            if bound != "*":
                record(bound)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            record(node.name)
        elif isinstance(node, ast.ExceptHandler):
            record(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)):
            record(node.name)
        elif isinstance(node, ast.MatchMapping):
            record(node.rest)

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    candidates[target.id] = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            candidates[node.target.id] = node.value

    return {
        name: value for name, value in candidates.items()
        if counts.get(name) == 1
    }


def _direct_data_binding_names(tree: ast.AST) -> set[str]:
    """Names assigned a `data/...` literal at least once in this tree."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_data_path(node.value):
            names.update(
                target.id for target in node.targets
                if isinstance(target, ast.Name)
            )
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
            and _is_data_path(node.value)
        ):
            names.add(node.target.id)
    return names


# Callables whose argument is a path being touched. This longer list does not
# cost precision because the argument itself — or its unique local binding —
# remains `_is_data_path`-gated. Thus `json.load(fh)` and
# `torch.load(state)` can never become opened hits.
_FS_CALLS = frozenset({
    "Path",
    "exists",
    "glob",
    "iterdir",
    "listdir",
    "load",
    "loadtxt",
    "makedirs",
    "mkdir",
    "open",
    "read_bytes",
    "read_csv",
    "read_json",
    "read_parquet",
    "read_pickle",
    "read_text",
    "rglob",
    "save",
    "savez",
    "unlink",
    "write_bytes",
    "write_text",
})


def _data_path_literal_analysis_in(path: Path) -> tuple[dict[str, int], int]:
    """Return the two literal buckets and ambiguous local-name count."""
    split = {"opened": 0, "inert": 0}
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError, OSError):
        return split, 0

    bindings = _single_bindings(tree)
    data_binding_names = _direct_data_binding_names(tree)
    in_call: set[int] = set()
    unresolved_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        called = (
            func.id if isinstance(func, ast.Name)
            else func.attr if isinstance(func, ast.Attribute)
            else ""
        )
        if called not in _FS_CALLS:
            continue
        for arg in [*node.args, *(kw.value for kw in node.keywords)]:
            if _is_data_path(arg):
                in_call.add(id(arg))
                continue
            if not isinstance(arg, ast.Name):
                continue
            bound = bindings.get(arg.id)
            if bound is not None and _is_data_path(bound):
                in_call.add(id(bound))
            elif arg.id in data_binding_names:
                unresolved_names.add(arg.id)

    for node in ast.walk(tree):
        if not _is_data_path(node):
            continue
        split["opened" if id(node) in in_call else "inert"] += 1
    return split, len(unresolved_names)


def data_path_literals_in(path: Path) -> dict[str, int]:
    """One file's `data/...` literals, split by where each one sits.

    `opened`: the literal is a direct argument, or the unique local binding of
    an argument, to a call named in `_FS_CALLS`. `inert`: its syntax does not
    put it in a call position this guard recognises. Neither bucket establishes
    what the file does at runtime.
    """
    return _data_path_literal_analysis_in(path)[0]


def unresolved_data_path_names_in(path: Path) -> int:
    """Locally data-bound call arguments rejected as non-single bindings."""
    return _data_path_literal_analysis_in(path)[1]


def data_path_literals() -> dict[str, dict[str, int]]:
    """Per core module, its `data/...` string literals split by kind.

    Literals shaped like a repo-relative data path, classified syntactically as
    sitting in a filesystem-call position (`opened`) or not (`inert`). The
    origin of a path received as a PARAMETER is invisible to this single-file
    binding measure — see the module docstring.
    """
    counts: dict[str, dict[str, int]] = {}
    modules = module_table()
    for name in sorted(core_modules()):
        split = data_path_literals_in(modules[name])
        if split["opened"] or split["inert"]:
            counts[name] = split
    return counts


def unresolved_data_path_names() -> int:
    """Ambiguous local data-path names at recognised calls across the core."""
    modules = module_table()
    return sum(
        unresolved_data_path_names_in(modules[name])
        for name in core_modules()
    )


def _div_segments(node: ast.AST) -> list[ast.AST]:
    """Flatten one `/` expression from left to right."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return [*_div_segments(node.left), *_div_segments(node.right)]
    return [node]


def _is_resolved_file_call(node: ast.AST) -> bool:
    """Exact `Path(__file__).resolve()` syntax."""
    if (
        not isinstance(node, ast.Call)
        or node.args
        or node.keywords
        or not isinstance(node.func, ast.Attribute)
        or node.func.attr != "resolve"
    ):
        return False
    path_call = node.func.value
    return (
        isinstance(path_call, ast.Call)
        and not path_call.keywords
        and isinstance(path_call.func, ast.Name)
        and path_call.func.id == "Path"
        and len(path_call.args) == 1
        and isinstance(path_call.args[0], ast.Name)
        and path_call.args[0].id == "__file__"
    )


def _resolved_file_base(
    node: ast.AST,
    path: Path,
    bindings: Mapping[str, ast.AST],
    seen: frozenset[str] = frozenset(),
) -> Path | None:
    """Compute only explicit `__file__` parent syntax; never guess a base."""
    if isinstance(node, ast.Name):
        if node.id in seen or node.id not in bindings:
            return None
        return _resolved_file_base(
            bindings[node.id], path, bindings, seen | {node.id}
        )

    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "parents"
        and _is_resolved_file_call(node.value.value)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, int)
        and not isinstance(node.slice.value, bool)
        and node.slice.value >= 0
    ):
        try:
            return path.resolve().parents[node.slice.value]
        except IndexError:
            return None

    parent_steps = 0
    cursor = node
    while isinstance(cursor, ast.Attribute) and cursor.attr == "parent":
        parent_steps += 1
        cursor = cursor.value
    if not _is_resolved_file_call(cursor):
        return None
    base = path.resolve()
    for _ in range(parent_steps):
        base = base.parent
    return base


def _assembled_base(
    segments: list[ast.AST],
    data_index: int,
    path: Path,
    bindings: Mapping[str, ast.AST],
) -> str:
    """Classify the syntactically computable prefix before `/ "data"`."""
    prefix = segments[:data_index]
    if not prefix:
        return "unresolved base"
    base = _resolved_file_base(prefix[0], path, bindings)
    if base is None:
        return "unresolved base"
    for segment in prefix[1:]:
        if not (
            isinstance(segment, ast.Constant)
            and isinstance(segment.value, str)
        ):
            return "unresolved base"
        base /= segment.value
    resolved = base.resolve()
    if resolved == ROOT.resolve():
        return "repo data/"
    if resolved == (SRC / PACKAGE).resolve():
        return "PACKAGE data"
    return "unresolved base"


def assembled_data_paths() -> list[dict]:
    """Maximal `/` chains with a `"data"` segment in core modules.

    The expression and line are derived from the live AST. Base classification
    accepts only computable `Path(__file__).resolve()` parent forms, directly or
    through one unique local binding; every other prefix stays unresolved.
    """
    sites: list[dict] = []
    modules = module_table()
    for name in sorted(core_modules()):
        path = modules[name]
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        bindings = _single_bindings(tree)
        parents = {
            id(child): parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
                continue
            parent = parents.get(id(node))
            if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Div):
                continue
            segments = _div_segments(node)
            data_indexes = [
                index for index, segment in enumerate(segments)
                if (
                    isinstance(segment, ast.Constant)
                    and isinstance(segment.value, str)
                    and segment.value == "data"
                )
            ]
            if not data_indexes:
                continue
            data_index = data_indexes[0]
            sites.append({
                "module": name,
                "line": node.lineno,
                "expr": ast.unparse(node),
                "base": _assembled_base(
                    segments, data_index, path, bindings
                ),
            })
    sites.sort(key=lambda site: (site["module"], site["line"], site["expr"]))
    return sites


def coverage() -> dict:
    """What the guard read. Honesty, not correctness.

    `closure` is the serving closure and `tool_closure` the script one; `unreached`
    is core minus BOTH of them — modules no serving entry point and no script
    reaches — and `unreached_modules` names them. `undisposed` counts the ones
    with nothing written about them, which is a gap in the prose, not in the tree.
    `data_path_literals_*` are counts of literals, not of reads;
    `unresolved_data_path_names` counts rejected ambiguous bindings, and
    `assembled_data_paths` counts a separate syntactic category.
    """
    graph = import_graph()
    core = core_modules()
    checked = [edge for name in core for edge in graph[name]]
    split = unreached_modules()
    literals = data_path_literals()
    return {
        "modules": len(module_table()),
        "core": len(core),
        "research": len(research_modules()),
        "closure": len(core_closure()),
        "tool_closure": len(tool_closure()),
        "tool_only": len(split["tool_only"]),
        "unreached": len(split["unreached"]),
        "unreached_modules": split["unreached"],
        "undisposed": len(undisposed_modules()),
        "edges_checked": len(checked),
        "edges_module_level": sum(1 for e in checked if e.scope == "module"),
        "edges_function_level": sum(1 for e in checked if e.scope == "function"),
        "entry_points": len(serving_entry_points()),
        "tool_entry_points": sum(1 for e in tool_entry_points().values() if e),
        "dynamic_import_sites": len(dynamic_import_sites()),
        "data_path_literals": sum(sum(v.values()) for v in literals.values()),
        "data_path_literals_in_call_position": sum(
            v["opened"] for v in literals.values()
        ),
        "unresolved_data_path_names": unresolved_data_path_names(),
        "assembled_data_paths": len(assembled_data_paths()),
    }


def main(argv: list | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    counts = coverage()
    print(
        f"Scanning {counts['modules']} first-party module(s) under "
        f"src/{PACKAGE}/ for core -> research imports:"
    )
    print(f"  serving entry points ({counts['entry_points']}):")
    for name in serving_entry_points():
        why = DECLARED_ENTRY_POINTS.get(name, "reached from the package Public API")
        print(f"    {name} — {why.split('.')[0]}.")
    print(f"  research roots ({len(RESEARCH_ROOTS)}):")
    for root in RESEARCH_ROOTS:
        print(f"    {root}/ — {RESEARCH_ROOTS[root].split('.')[0]}.")

    stale = stale_declarations()
    if stale:
        print(
            f"\nSTALE DECLARATIONS — {len(stale)} entry(ies) in "
            f"{Path(__file__).name} no longer describe the tree or its "
            f"reachability. Coverage cannot be reported against a table that "
            f"has rotted:\n"
        )
        for item in stale:
            print(f"  {item['kind']}: {item['name']}")
        return 3

    violations = find_boundary_violations()
    if violations:
        print(
            f"\nBOUNDARY VIOLATIONS — {len(violations)} import(s) run core -> "
            f"research. The core must be deployable without the harness; invert "
            f"the dependency or move the shared piece down into the core:\n"
        )
        for v in violations:
            where = "serving-reachable" if v["serving_reachable"] else "not on a serving path"
            print(
                f"  {v['importer']} -> {v['target']}  "
                f"({v['file']}, {v['scope']}-level import, {where}; research root "
                f"{v['root']})"
            )
        return 1

    holes = dynamic_import_sites()
    literals = data_path_literals()
    assembled = assembled_data_paths()
    split = unreached_modules()
    print(
        f"\nCLEAN — no core module imports the research harness. Checked "
        f"{counts['edges_checked']} first-party import(s) "
        f"({counts['edges_module_level']} module-level, "
        f"{counts['edges_function_level']} inside functions) across "
        f"{counts['core']} core module(s); {counts['research']} module(s) are "
        f"declared research and are not checked."
    )
    print(
        f"  Serving closure: {counts['closure']} module(s) reachable from the "
        f"entry points. Tool closure ({counts['tool_entry_points']} file(s) under "
        f"{TOOL_ENTRY_DIR.name}/ import {PACKAGE}): {counts['tool_closure']} "
        f"module(s) — a separate claim, not a serving path."
    )
    reachers = test_reachers()
    print(
        f"  Off the serving path: {counts['tool_only']} core module(s) are reached "
        f"only by a script, and {counts['unreached']} by no serving entry point "
        f"and no file under {TOOL_ENTRY_DIR.name}/ — the two root sets this guard "
        f"declares. Both are guarded by default, not by closure. That second "
        f"number is a question this file answers per module below, not a verdict: "
        f"an import graph shows that nothing HERE imports a module, never that "
        f"nothing needs it."
    )
    for module, scripts in sorted(split["tool_only"].items()):
        via = ", ".join(f"{TOOL_ENTRY_DIR.name}/{s}" for s in scripts)
        print(f"    script-reached only: {module} (via {via})")
    for module in split["unreached"]:
        via = ", ".join(f"{TESTS_ENTRY_DIR.name}/{t}" for t in reachers.get(module, ()))
        covered = f" imported by {via};" if via else ""
        why = UNREACHED_DISPOSITIONS.get(module)
        reason = (
            _first_sentence(why) if why else
            "UNDISPOSED — no entry in UNREACHED_DISPOSITIONS "
            f"({Path(__file__).name}) says what this module is or why it stays. "
            "Write one; a missing answer is not a delete instruction."
        )
        print(f"    no closure here reaches: {module} —{covered} {reason}")
    print(
        f"  Cannot see: {counts['dynamic_import_sites']} dynamic import site(s). "
        f"Of {counts['data_path_literals']} `data/...` string literal(s) in "
        f"{len(literals)} core module(s), "
        f"{counts['data_path_literals_in_call_position']} sit in a "
        f"filesystem-call position and "
        f"{counts['data_path_literals'] - counts['data_path_literals_in_call_position']}"
        f" do not sit in any call position this guard recognises — which "
        f"establishes that and nothing more; a literal can be inert here and "
        f"still name a real dependency expressed another way."
    )
    for module, kinds in sorted(literals.items(), key=lambda kv: (-sum(kv[1].values()), kv[0])):
        print(
            f"    {module}: {kinds['opened']} in a filesystem-call position, "
            f"{kinds['inert']} inert"
        )
    print(
        f"  Local indirection: {counts['unresolved_data_path_names']} "
        f"data-path name(s) used in a recognised call could not be resolved "
        f"because the name was not single-bound."
    )
    print(
        f"  Assembled data paths: {counts['assembled_data_paths']} maximal "
        f"`/` chain(s) contain a `data` segment:"
    )
    for site in assembled:
        print(
            f"    {site['module']} line {site['line']}: {site['expr']} — "
            f"{site['base']}"
        )
    # STATED POSITIVELY, ON PURPOSE. Four rounds of this guard tried to enumerate
    # what it CANNOT see, and every round shipped an incomplete list — the last one
    # named `os.path.join` assembly but not `/`-operator assembly, so
    # `open(ROOT / "data/x.jsonl")` scored `inert` while the sentence above invited
    # the reader to think that shape was covered. A blacklist of blind spots is not
    # completable; the recognised set is. Enumerate THAT and let everything else
    # fall outside by construction.
    print(
        f"  What the literal measure recognises, exhaustively: a `data/...` string "
        f"literal appearing as a bare argument, or as a bare local name bound "
        f"exactly once to such a literal, in a call to one of {len(_FS_CALLS)} names "
        f"({', '.join(sorted(_FS_CALLS))}). The `/`-chain measure above recognises a "
        f"chain with a segment exactly equal to 'data'."
    )
    print(
        "  EVERY other expression shape is unmeasured and falls to `inert` by "
        "default — any other operand of `/`, `.joinpath`, an f-string, "
        "`os.path.join`, a container or attribute lookup, a name bound twice, a "
        "parameter, or a value from another module. `inert` means 'not in a shape "
        "these measures read', NEVER 'not a read'."
    )
    print(
        "  None of that covers a path taken as a PARAMETER, which the guard "
        "cannot see at all: src/indra_belief/calibration_constants.py:189 opens "
        "whatever `run_path` its caller hands it. A count of 0 literals in a "
        "call position is not a finding of no path coupling."
    )
    for hole in holes:
        print(f"    dynamic import: {hole['module']} line {hole['line']} ({hole['call']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
