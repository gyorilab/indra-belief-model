"""Guard the deployable core against imports from the research harness.

The repository holds a deployable core and a research harness that measures it.
Research inherits the core. The reverse edge is the defect: a core module that
imports the harness cannot be deployed without it.

Only research is declared. `RESEARCH_ROOTS` is the single declaration, while
`core_modules()` is its complement over `module_table()`'s source-tree glob.
Checked is therefore the default: a module added under `src/indra_belief/` is
checked on the commit that adds it without an edit here. Adding a prefix to
`RESEARCH_ROOTS` removes coverage, so every such prefix has to earn the exemption.

Function-level imports are edges. `import_edges` walks function bodies, so a
deferred import is recorded rather than excused, and the clean report prints the
module-level and function-level split.

Exit codes:
    0  no core module imports a research module
    1  a core -> research import
    3  a research-root declaration no longer names a module

The repairs differ in kind. Exit 1 means move code or invert a dependency. Exit
3 means edit the declaration or restore the module it names; continuing with a
rotted declaration could quietly narrow the rule until it enforces nothing.

This is an import-graph check and nothing more. It reads only imports written in
source. Dynamic imports, paths taken as parameters, callbacks, registries, and
string dispatch are invisible. A green run does not mean the core is
deployable. For example,
`calibration_constants::_call_log_fingerprints` opens whatever run file it is
handed; no literal at the call site says which artifact that is. These examples
state the check's scope, not a complete enumeration of possible blind spots.

Pytest reaches this guard through `tests/test_import_boundary.py`, mirroring how
`scripts/check_contamination.py` is reached through
`tests/test_contamination_guard_sources.py`.

In CI, this guard is reached only through the `python -m pytest -q` step in
`.github/workflows/ci.yml`; it has no workflow step of its own.
"""
from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGE = "indra_belief"

# The measurement side. A module under one of these prefixes exists to say
# whether a belief is good, not to produce one, and the core may not import it.
# Keep this short: everything NOT under a root here is checked, so a prefix added
# here removes guard coverage and has to earn it.
RESEARCH_ROOTS: Mapping[str, str] = {
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


def find_boundary_violations(
    graph: Mapping[str, tuple[Edge, ...]] | None = None,
) -> list[dict]:
    """Every import from a core module into a research module.

    Each record is {"importer", "target", "line", "scope", "root", "file"}.
    An empty list means the core imports nothing from the research harness.
    Importable so the pytest guard can call it directly.
    """
    graph = import_graph() if graph is None else graph
    modules = module_table()
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
                "root": root,
                "file": str(modules[name].relative_to(ROOT)),
            })
    violations.sort(key=lambda v: (v["importer"], v["target"], v["line"]))
    return violations


def stale_declarations() -> list[dict]:
    """Research roots that no longer name any first-party module.

    If the harness moves out from under a declared prefix, those modules silently
    become core. Core -> core imports are legal, so the guard can keep exiting 0
    while enforcing nothing. A research root naming no module is therefore exit
    3, not a warning.
    """
    modules = module_table()
    stale: list[dict] = []
    for root in RESEARCH_ROOTS:
        if root not in modules and not any(
            m.startswith(root + ".") for m in modules
        ):
            stale.append({"kind": "research root", "name": root})
    return stale


def coverage() -> dict:
    """Counts for the import-graph rule and its clean report.

    Edges are counted from derived core modules and split by module/function
    scope so deferred imports remain visible.
    """
    graph = import_graph()
    core = core_modules()
    checked = [edge for name in core for edge in graph[name]]
    return {
        "modules": len(module_table()),
        "core": len(core),
        "research": len(research_modules()),
        "edges_checked": len(checked),
        "edges_module_level": sum(1 for e in checked if e.scope == "module"),
        "edges_function_level": sum(1 for e in checked if e.scope == "function"),
    }


def main(argv: list | None = None) -> int:
    counts = coverage()
    print(
        f"Scanning {counts['modules']} first-party module(s) under "
        f"src/{PACKAGE}/ for core -> research imports:"
    )
    print(f"  research roots ({len(RESEARCH_ROOTS)}):")
    for root in RESEARCH_ROOTS:
        print(f"    {root}/ — {RESEARCH_ROOTS[root].split('.')[0]}.")

    stale = stale_declarations()
    if stale:
        print(
            f"\nSTALE DECLARATIONS — {len(stale)} research root(s) in "
            f"{Path(__file__).name} no longer name a module. The guard will not "
            f"silently narrow its coverage:\n"
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
            print(
                f"  {v['importer']} -> {v['target']} "
                f"({v['file']}, {v['scope']}-level import, research root "
                f"{v['root']})"
            )
        return 1

    print(
        f"\nCLEAN — no core module imports the research harness. Checked "
        f"{counts['edges_checked']} first-party import(s) "
        f"({counts['edges_module_level']} module-level, "
        f"{counts['edges_function_level']} function-level) across "
        f"{counts['core']} core module(s); {counts['research']} module(s) are "
        f"declared research and are not checked."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
