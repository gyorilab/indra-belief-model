"""The core/research boundary, and proof the guard that holds it can fail.

`scripts/check_import_boundary.py` asserts that the deployable core imports
nothing from the research harness. Most tests below construct a real inversion
or a rotted declaration and require the guard to catch it; the live clean-tree
assertion alone would pass against a guard that always returned success.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Register before exec: the module defines a @dataclass, whose decorator resolves
# `cls.__module__` through sys.modules and raises if the name is not yet there.
_spec = importlib.util.spec_from_file_location(
    "check_import_boundary", ROOT / "scripts" / "check_import_boundary.py"
)
cib = importlib.util.module_from_spec(_spec)
sys.modules["check_import_boundary"] = cib
_spec.loader.exec_module(cib)


def test_the_core_imports_no_research_module():
    """The live invariant. On its own this proves nothing about the guard."""
    assert cib.find_boundary_violations() == []


def test_no_declaration_has_gone_stale():
    """Every declared research root must still name at least one module.

    If the harness moves out from under a declared prefix, its modules silently
    become core and their core-to-core imports become legal. A root that names
    nothing would let the guard exit 0 while enforcing nothing.
    """
    assert cib.stale_declarations() == []


def test_a_module_added_under_src_is_checked_without_editing_the_guard(
    monkeypatch, tmp_path,
):
    """The anti-rot property, and the only reason `core` is derived and not listed.

    A hand-maintained core list passes on the day it is written and rots on the
    next commit. This proves the opposite shape: a module whose name appears
    nowhere in the guard is checked because it EXISTS under `src/indra_belief/`,
    and its import into a declared research root is caught with no edit to the
    guard. Both halves matter — the first is the derivation, the second is the rule.
    """
    pkg = tmp_path / "indra_belief"
    (pkg / "results").mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "results" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "brand_new_module.py").write_text(
        "from indra_belief.results import anything\n", encoding="utf-8"
    )
    monkeypatch.setattr(cib, "SRC", tmp_path)
    # The violation record remains relative to ROOT; keep that production lookup
    # unchanged by pointing its synthetic root at the same temporary tree.
    monkeypatch.setattr(cib, "ROOT", tmp_path)
    cib.module_table.cache_clear()
    cib.import_graph.cache_clear()
    try:
        assert f"{cib.PACKAGE}.brand_new_module" in cib.core_modules()
        found = cib.find_boundary_violations()
        assert any(
            v["importer"] == f"{cib.PACKAGE}.brand_new_module"
            and v["target"] == f"{cib.PACKAGE}.results"
            for v in found
        ), found
    finally:
        monkeypatch.undo()
        cib.module_table.cache_clear()
        cib.import_graph.cache_clear()


@pytest.mark.parametrize("importer", ["prepared_execution", "verdict"])
def test_a_seeded_core_to_research_import_is_caught(importer):
    """THE GATE. Seed the inversion the guard exists to catch.

    Injected into an in-memory graph rather than written to disk, so a crashed
    test cannot leave a real inversion behind in a shared checkout.
    """
    graph = dict(cib.import_graph())
    assert cib.find_boundary_violations(graph) == []

    name = f"{cib.PACKAGE}.{importer}"
    # `comparison.contracts` was the exemplar until the comparison harness was
    # removed. Any declared research root serves: the guard checks direction.
    target = f"{cib.PACKAGE}.results"
    graph[name] = tuple(graph[name]) + (
        cib.Edge(importer=name, target=target, line=1, scope="module"),
    )

    found = cib.find_boundary_violations(graph)
    assert any(v.get("importer") == name and v.get("target") == target for v in found), (
        f"the guard did not catch {name} -> {target}; got {found}"
    )


def test_a_lazy_import_inside_a_function_is_not_a_loophole(tmp_path):
    """An import nested in a function is the usual way a boundary gets crossed quietly.

    A guard reading only module-level imports would bless it, so the AST walk must
    reach it AND record that it was function-scoped.
    """
    src = tmp_path / "sneaky.py"
    src.write_text(
        "def score(record):\n"
        "    from indra_belief.results import build_run_export\n"
        "    return build_run_export\n",
        encoding="utf-8",
    )
    edges = cib.import_edges(f"{cib.PACKAGE}.sneaky", src)
    targets = {e.target for e in edges}
    assert any(t.startswith(f"{cib.PACKAGE}.results") for t in targets), targets
    assert any(e.scope != "module" for e in edges), (
        "a function-scoped import must not be recorded as module-level"
    )


def test_contract_error_lives_in_the_core_and_still_binds_one_object():
    """`ContractError` belongs to the core and `ReplayError` derives from it.

    This test began as a fork check. `ContractError` was moved into the core to
    repair the one back-edge this guard exists to prevent, and
    `comparison.contracts` re-exported it; two distinct classes would have meant
    every `except ContractError` in the harness silently stopped catching the
    core's error. The harness has since been removed outright, so there is no
    second importer left to fork FROM — what survives, and is still worth
    pinning, is where the class lives and what it derives from.
    """
    from indra_belief.prepared_execution import ContractError as core_side
    from indra_belief.prepared_execution import ReplayError

    assert issubclass(core_side, ValueError)
    assert issubclass(ReplayError, core_side)


def test_a_research_root_that_names_nothing_is_a_stale_declaration(monkeypatch):
    """A rotted research prefix fails loudly instead of quietly removing coverage."""
    missing = f"{cib.PACKAGE}.research_root_moved_away"
    monkeypatch.setattr(
        cib,
        "RESEARCH_ROOTS",
        {missing: "test-only stale declaration"},
    )
    cib.module_table.cache_clear()
    cib.import_graph.cache_clear()
    try:
        stale = cib.stale_declarations()
        assert {"kind": "research root", "name": missing} in stale, stale
        assert cib.main([]) == 3
    finally:
        monkeypatch.undo()
        cib.module_table.cache_clear()
        cib.import_graph.cache_clear()


def test_the_clean_report_prints_the_split_it_derives(capsys):
    """main()'s exit-0 branch is executed by something other than a human.

    The trim deleted the report machinery and, with it, the only test that ran
    ``main()`` past an early return. CI reaches this guard solely through
    ``python -m pytest -q`` and never invokes the script, so without this the
    CLEAN paragraph and every f-string key in it were enforced by nothing: a
    renamed ``coverage()`` key would leave the suite green and surface only as
    a KeyError at the terminal of whoever next ran the guard by hand.

    The expected numbers are DERIVED from ``coverage()`` rather than pinned, so
    this asserts the report says what it measured, not that it says 62.
    """
    counts = cib.coverage()
    assert cib.main([]) == 0
    out = capsys.readouterr().out
    assert "CLEAN" in out, out
    for key in ("edges_checked", "edges_module_level", "edges_function_level", "core"):
        assert str(counts[key]) in out, f"{key}={counts[key]} missing from:\n{out}"


def test_a_violation_is_reported_with_the_fields_needed_to_fix_it(monkeypatch, capsys):
    """main()'s exit-1 branch names importer, target, file, scope and root.

    A guard that detects a violation but prints an unactionable line is only
    half a guard, and nothing else executes this format.
    """
    monkeypatch.setattr(
        cib,
        "find_boundary_violations",
        lambda: [
            {
                "importer": f"{cib.PACKAGE}.verdict",
                "target": f"{cib.PACKAGE}.results",
                "file": f"src/{cib.PACKAGE}/verdict.py",
                "scope": "module",
                "root": f"{cib.PACKAGE}.results",
            }
        ],
    )
    try:
        assert cib.main([]) == 1
        out = capsys.readouterr().out
        assert "BOUNDARY VIOLATIONS" in out, out
        for field in ("verdict", "results", "verdict.py", "module-level"):
            assert field in out, f"{field!r} missing from:\n{out}"
    finally:
        monkeypatch.undo()
