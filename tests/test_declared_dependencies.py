"""Every third-party module the suite imports must be DECLARED in pyproject.toml.

WHAT THIS CATCHES, and it caught it in production rather than in theory.
`tests/test_frontier_plot.py` imports `scripts/frontier_plot.py`, which imports
plotly at module level. plotly was in nobody's dependency list, so `pip install
-e ".[dev]"` on a clean runner did not install it and pytest died at COLLECTION
with ModuleNotFoundError — before running a single test, and therefore before
every CI step that follows the test step.

WHY IT SURVIVED A FRESH-CHECKOUT REPRODUCTION. The obvious way to reproduce CI
locally is `git worktree add`, which gives a tree holding only tracked files.
That reproduces the FILE TREE and reuses the developer's existing virtualenv, so
a dependency that is installed here and declared nowhere is invisible to it. The
only faithful reproduction is a clean interpreter plus the declared extras —
which is what this test approximates statically, on every machine, in a second.

WHAT IT DOES NOT CLAIM, stated because the scope is where a guard like this
hides things.

  * MODULE LEVEL ONLY. A module imported inside a function body is invisible
    here and will still fail at call time. That is deliberate: the failure this
    exists to stop is a COLLECTION error, and collection executes module level
    and nothing deeper. Widening would flag every optional-backend import.
  * WHAT PYTEST REACHES, not the whole repo. Every file under `tests/`, plus the
    `scripts/` modules those files import, transitively. A script no test
    imports cannot break collection, and `scripts/run_indra_paper_literal_models.py`
    is the live example: it imports `bioexp`, the 2023 assembly paper's own
    package, which is a GitHub repository rather than a PyPI distribution.
    Declaring it would make every clean install clone it to run a script CI
    never runs. It is named here so the omission is a decision on the record and
    not a gap — and `test_the_scope_is_what_pytest_collects` pins the rule, so a
    test that starts importing that script pulls it back into scope.
  * TRANSITIVITY IS INVISIBLE. `scipy` arrives via scikit-learn; the repo pins
    it anyway so the floor is stated rather than inherited.
"""
from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Import name -> distribution name, where they differ. Kept tiny on purpose: a
# long table here would be a second place to maintain the dependency list, which
# is the defect this test exists to prevent.
_DISTRIBUTION_NAMES = {"sklearn": "scikit-learn"}

# Not third-party: the package under test, and the local sibling modules that
# `sys.path` manipulation in tests/ and scripts/ makes importable by bare name.
_FIRST_PARTY = {"indra_belief", "scripts", "tests"}


def _declared() -> set[str]:
    """Every distribution named in pyproject, across all dependency groups."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = config["project"]
    requirements = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)
    names = set()
    for requirement in requirements:
        # "pkg>=1.2" / "pkg[extra]>=1" / "pkg" -> "pkg"
        name = requirement.split(";")[0].strip()
        for separator in ("[", ">", "<", "=", "!", "~", " "):
            name = name.split(separator)[0]
        names.add(name.strip().lower().replace("_", "-"))
    return names


def _module_level_imports(path: Path) -> set[str]:
    """Top-level import names only — collection executes nothing deeper.

    Both the ROOT package and the `scripts.<name>` second segment are returned,
    because this repo reaches its scripts BOTH ways: `import frontier_plot`
    after a `sys.path` insert, and `from scripts.reproduce_... import ...`. A
    walker that kept only the first segment saw the second form as the local
    name `scripts` and never followed it — which silently shrank the closure to
    almost nothing while still reporting a clean result.
    """
    names = set()
    for node in ast.parse(path.read_text(), filename=str(path)).body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                names.add(parts[0])
                if parts[0] == "scripts" and len(parts) > 1:
                    names.add(parts[1])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            parts = node.module.split(".")
            names.add(parts[0])
            if parts[0] == "scripts" and len(parts) > 1:
                names.add(parts[1])
    return names


def _local_module_names() -> set[str]:
    """Bare names that resolve to a file in this repo rather than to a package."""
    return {
        path.stem
        for directory in ("scripts", "tests")
        for path in (ROOT / directory).glob("*.py")
    }


def collection_closure() -> dict[Path, set[str]]:
    """Files pytest imports at collection, mapped to their module-level imports.

    Every `tests/*.py`, plus each `scripts/*.py` a test imports, followed
    transitively — a script that imports another script is reached too.
    """
    scripts = {path.stem: path for path in (ROOT / "scripts").glob("*.py")}
    pending = list((ROOT / "tests").glob("*.py"))
    reached: dict[Path, set[str]] = {}
    while pending:
        path = pending.pop()
        if path in reached:
            continue
        names = _module_level_imports(path)
        reached[path] = names
        for name in names:
            if name in scripts and scripts[name] not in reached:
                pending.append(scripts[name])
    return reached


def test_every_module_level_import_pytest_reaches_is_declared():
    """The whole claim: a clean `pip install -e ".[dev]"` can collect the suite."""
    stdlib = set(sys.stdlib_module_names)
    local = _local_module_names() | _FIRST_PARTY
    declared = _declared()

    undeclared: dict[str, set[str]] = {}
    for path, names in collection_closure().items():
        for name in names:
            if name in stdlib or name in local or name.startswith("_"):
                continue
            distribution = _DISTRIBUTION_NAMES.get(name, name).lower().replace("_", "-")
            if distribution not in declared:
                undeclared.setdefault(distribution, set()).add(
                    str(path.relative_to(ROOT))
                )

    assert undeclared == {}, (
        "these third-party modules are imported at module level but declared in no "
        "pyproject dependency group, so a clean install cannot collect the suite: "
        + "; ".join(
            f"{name} (in {', '.join(sorted(files))})"
            for name, files in sorted(undeclared.items())
        )
    )


def test_the_check_can_fail(tmp_path):
    """A guard that cannot fail is not a guard.

    `test_every_module_level_import_in_tests_and_scripts_is_declared` would pass
    just as happily against an `_module_level_imports` that returned nothing.
    """
    probe = tmp_path / "probe.py"
    probe.write_text("import a_package_nobody_declares\nfrom another import thing\n")
    assert _module_level_imports(probe) == {"a_package_nobody_declares", "another"}


def test_a_function_body_import_is_deliberately_out_of_scope(tmp_path):
    """Stated as a limit rather than left for someone to discover.

    Collection executes module level and nothing deeper, so a lazy import cannot
    produce the ModuleNotFoundError-at-collection failure this guards. Widening
    to function bodies would flag every optional-backend import in the tree.
    """
    probe = tmp_path / "probe.py"
    probe.write_text("def f():\n    import lazily_imported\n")
    assert _module_level_imports(probe) == set()


def test_the_scope_is_what_pytest_collects():
    """The scope rule, pinned — so narrowing it later is a visible edit.

    A guard that quietly shrinks its own reach reports green for a smaller and
    smaller claim. Three properties hold it still: every test file is in scope,
    a script a test imports IS reached, and a script no test imports is NOT.
    """
    reached = collection_closure()
    test_files = set((ROOT / "tests").glob("*.py"))
    assert test_files <= set(reached), sorted(
        str(p.name) for p in test_files - set(reached)
    )

    # Reached, because tests/test_modularity_baseline.py loads it.
    assert ROOT / "scripts" / "reproduce_published_statement_beliefs.py" in reached

    # NOT reached: no test imports it. This is the `bioexp` case the module
    # docstring records — if a test ever imports this script, it enters the
    # closure and its undeclared imports start failing the check above.
    unreached = ROOT / "scripts" / "run_indra_paper_literal_models.py"
    assert unreached not in reached, (
        "a test now imports this script, so its dependencies (bioexp, a GitHub "
        "package rather than a PyPI distribution) are in scope and must be "
        "declared or the import made lazy"
    )


def test_the_declared_set_is_parsed_from_every_group():
    """Runtime deps and BOTH extras count — a dev-only tool is still declared."""
    declared = _declared()
    assert {"gilda", "indra", "numpy", "scikit-learn"} <= declared  # runtime
    assert {"openai", "anthropic"} <= declared                      # llm extra
    assert {"pytest", "plotly"} <= declared                         # dev extra
