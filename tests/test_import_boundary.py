"""The core/research boundary, and proof the guard that holds it can fail.

`scripts/check_import_boundary.py` asserts that the deployable core imports
nothing from the research harness. A guard is worth exactly what its failure mode
is worth, so most of this file seeds real inversions and requires the guard to
catch them. The single test that merely asserts today's tree is clean would pass
just as happily against a guard that returned 0 unconditionally.
"""
from __future__ import annotations

import ast
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
    """A declared entry point or research root that no longer resolves.

    This is the failure that makes every other check vacuous: rename a module,
    leave the declaration stale, and the closure quietly stops covering the thing
    you renamed while the guard still exits 0 over a smaller core nobody reads.
    """
    assert cib.stale_declarations() == []


def test_the_core_is_a_closure_over_imports_not_a_hand_list():
    """The core must FOLLOW imports, or it rots the way a hand-maintained list does.

    Proven by construction: the closure contains the kernels the entry points
    import and excludes the harness — and severing an edge must SHRINK it. A
    hand-written set would satisfy the first half by coincidence and the second
    half not at all.
    """
    core = cib.core_closure()
    assert f"{cib.PACKAGE}.prepared_execution" in core
    assert f"{cib.PACKAGE}.verdict" in core
    assert not any(cib.is_research(name) for name in core)
    assert len(core) < len(cib.module_table()), (
        "a closure that swallows every module is not discriminating"
    )


@pytest.mark.parametrize("importer", ["prepared_execution", "verdict"])
def test_a_seeded_core_to_research_import_is_caught(importer):
    """THE GATE. Seed the inversion the guard exists to catch.

    Injected into an in-memory graph rather than written to disk, so a crashed
    test cannot leave a real inversion behind in a shared checkout.
    """
    graph = dict(cib.import_graph())
    assert cib.find_boundary_violations(graph) == []

    name = f"{cib.PACKAGE}.{importer}"
    target = f"{cib.PACKAGE}.comparison.contracts"
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
        "    from indra_belief.comparison.replay import ReplayIndex\n"
        "    return ReplayIndex\n",
        encoding="utf-8",
    )
    edges = cib.import_edges(f"{cib.PACKAGE}.sneaky", src)
    targets = {e.target for e in edges}
    assert any(t.startswith(f"{cib.PACKAGE}.comparison") for t in targets), targets
    assert any(e.scope != "module" for e in edges), (
        "a function-scoped import must not be recorded as module-level"
    )


def test_contract_error_moved_to_the_core_and_still_binds_one_object():
    """The back-edge repair must not have forked the exception.

    `ContractError` moved into the core and `comparison.contracts` re-exports it.
    Two distinct classes would mean every `except ContractError` and
    `pytest.raises(ContractError)` in the harness silently stops catching the
    core's error — a boundary fix that breaks error handling is not a fix.
    """
    from indra_belief.comparison.contracts import ContractError as harness_side
    from indra_belief.prepared_execution import ContractError as core_side
    from indra_belief.prepared_execution import ReplayError

    assert core_side is harness_side
    assert issubclass(core_side, ValueError)
    assert issubclass(ReplayError, core_side)


def test_the_guard_reports_what_it_cannot_see(tmp_path):
    """Coverage honesty. A guard that overstates its reach is worse than none.

    Two different things must both be reported. The dynamic-import hole stays on
    the screen even at zero. And the `data/...` literals must be reported as what
    they ARE: the core's ten literals are provenance records inside
    `calibration_constants`, not reads, and calling them "runtime reads of
    research artifacts" was the guard describing a coupling its own measure never
    established. The classifier below must therefore be a discriminator — a
    counter that says "opened" unconditionally fails the second assertion, and
    one that says "inert" unconditionally fails the third.
    """
    cov = cib.coverage()
    assert "dynamic_import_sites" in cov, cov

    literals = cib.data_path_literals()
    assert literals, "the core does carry data/ literals; reporting none understates"
    constants = literals[f"{cib.PACKAGE}.calibration_constants"]
    assert constants["inert"] and not constants["opened"], (
        f"those literals are provenance dict values, not reads: {constants}"
    )
    assert sum(v["opened"] for v in literals.values()) == 0, literals

    reader = tmp_path / "reader.py"
    reader.write_text("def load():\n    return open('data/x.jsonl').read()\n",
                      encoding="utf-8")
    assert cib.data_path_literals_in(reader) == {"opened": 1, "inert": 0}


@pytest.mark.parametrize(
    "source",
    [
        'P = "data/x.jsonl"\ndef load():\n    return open(P)\n',
        'def load():\n    P = "data/x.jsonl"\n    return open(P)\n',
    ],
    ids=["module-binding", "function-binding"],
)
def test_one_local_data_path_binding_is_resolved(tmp_path, source):
    """A single binding is followed at module and function scope."""
    reader = tmp_path / "indirect_reader.py"
    reader.write_text(source, encoding="utf-8")

    assert cib.data_path_literals_in(reader) == {"opened": 1, "inert": 0}


@pytest.mark.parametrize(
    "source",
    [
        'pd.read_csv("data/x.csv")\n',
        'np.load("data/w.npy")\n',
    ],
    ids=["pandas-read-csv", "numpy-load"],
)
def test_data_readers_beyond_builtin_open_are_recognised(tmp_path, source):
    """Reader names widen the measure while the path literal remains the gate."""
    reader = tmp_path / "library_reader.py"
    reader.write_text(source, encoding="utf-8")

    assert cib.data_path_literals_in(reader) == {"opened": 1, "inert": 0}


def test_widened_filesystem_calls_remain_data_literal_gated(tmp_path):
    """Handle/state arguments must not become path hits merely from call names."""
    reader = tmp_path / "non_path_loads.py"
    reader.write_text("json.load(fh)\ntorch.load(state)\n", encoding="utf-8")

    assert cib.data_path_literals_in(reader) == {"opened": 0, "inert": 0}
    assert cib.unresolved_data_path_names_in(reader) == 0


@pytest.mark.parametrize(
    ("source", "inert"),
    [
        ('P = "data/a.json"\nP = "data/b.json"\nopen(P)\n', 2),
        ('P = "data/a.json"\nP += ".gz"\nopen(P)\n', 1),
    ],
    ids=["assigned-twice", "augmented"],
)
def test_rebound_data_path_names_are_not_resolved_silently(tmp_path, source, inert):
    """Ambiguous names stay inert and increment the separately reported count."""
    reader = tmp_path / "rebound_reader.py"
    reader.write_text(source, encoding="utf-8")

    assert cib.data_path_literals_in(reader) == {"opened": 0, "inert": inert}
    assert cib.unresolved_data_path_names_in(reader) == 1


def test_an_assembled_data_path_is_derived_from_a_tmp_module(
    monkeypatch, tmp_path,
):
    """Module, line, and expression come from the supplied AST at runtime."""
    leaf = f"{tmp_path.name}.json.gz"
    source = (
        "BASE = object()\n"
        f'TARGET = BASE / "data" / "benchmark" / "{leaf}"\n'
    )
    path = tmp_path / "assembled.py"
    path.write_text(source, encoding="utf-8")
    module = f"{cib.PACKAGE}.synthetic_{tmp_path.name}"
    tree = ast.parse(source)
    assignment = next(
        node for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "TARGET"
        )
    )
    expression = assignment.value

    monkeypatch.setattr(cib, "module_table", lambda: {module: path})
    monkeypatch.setattr(cib, "core_modules", lambda: frozenset({module}))
    try:
        assert cib.assembled_data_paths() == [{
            "module": module,
            "line": expression.lineno,
            "expr": ast.unparse(expression),
            "base": "unresolved base",
        }]
    finally:
        monkeypatch.undo()


def test_live_assembled_bases_distinguish_repo_from_package_data():
    """Opposing live bases prevent a constant-returning classifier from passing."""
    sites = cib.assembled_data_paths()
    corpus_sites = [
        site for site in sites
        if (
            site["module"] == f"{cib.PACKAGE}.data.corpus"
            and "indra_benchmark_corpus.json.gz" in site["expr"]
        )
    ]
    package_sites = [
        site for site in sites
        if (
            site["module"] == f"{cib.PACKAGE}.scorers.monolithic.scorer"
            and "parents[2]" in site["expr"]
            and "example_bank.json" in site["expr"]
        )
    ]

    assert len(corpus_sites) == 1, corpus_sites
    assert len(package_sites) == 1, package_sites
    assert corpus_sites[0]["base"] == "repo data/", corpus_sites[0]
    assert package_sites[0]["base"] == "PACKAGE data", package_sites[0]
    assert corpus_sites[0]["base"] != package_sites[0]["base"]


def test_a_script_reached_module_is_not_unreached_and_the_reason_is_derived():
    """"No closure reaches it" must mean no closure, or it is not a signal.

    The panel package is imported by a build script, so a report that put it in
    the third bucket would be describing live tooling as unimported. The reacher
    is read off `scripts/` by the same AST that reads the package — no per-module
    table — which is why the assertion can name the script.

    `_prompts_verdict_only` USED to be the second example here, reached only by
    build_verdict_only_replay.py. Registering `verdict_only` in scorer.VARIANTS
    moved it into the serving closure: it is now part of what `import
    indra_belief` costs, which is exactly the transition this split exists to
    make visible. Asserted in its new bucket rather than deleted, so a future
    change that quietly drops it back out is still caught.

    The bucket itself is asserted as a DISCRIMINATOR and not as a copied list of
    three module names. A hardcoded triple here is precisely the hand-maintained
    list the guard's own docstring says rots: it passes for the wrong reason on
    the day someone adds a module, and it says nothing about whether the split
    discriminates. Non-empty and strictly inside the core does.
    """
    split = cib.unreached_modules()
    prompts = f"{cib.PACKAGE}.scorers.monolithic._prompts_verdict_only"

    assert prompts not in split["unreached"], split["unreached"]
    assert prompts in split["serving"], (
        "the verdict-only prompt is imported by scorer.VARIANTS, so it is "
        "serving-reachable; if it left that bucket the variant registration went "
        "away with it"
    )
    assert not (set(split["tool_only"]) & set(split["serving"])), (
        "a module reached by a serving entry point is not 'tool-reached only'"
    )

    unreached = split["unreached"]
    assert unreached, "an empty third bucket makes every claim about it vacuous"
    assert set(unreached) < set(cib.core_modules()), (
        "the unreached bucket is a strict subset of the core, never the core itself"
    )
    assert all(name in cib.UNREACHED_DISPOSITIONS for name in unreached), unreached


def test_an_emptied_script_root_is_a_stale_declaration(monkeypatch, tmp_path):
    """The derived glob must not silently empty.

    If `scripts/` moves, every script-reached module becomes an orphan and the
    report keeps exiting 0 over a claim it can no longer support. That is the same
    failure as a renamed entry point, and it takes the same exit code.
    """
    assert cib.stale_declarations() == []
    monkeypatch.setattr(cib, "TOOL_ENTRY_DIR", tmp_path / "gone")
    cib.tool_entry_points.cache_clear()
    cib.tool_closure.cache_clear()
    try:
        stale = cib.stale_declarations()
        assert any(item["kind"] == "tool root" for item in stale), stale
        assert cib.main([]) == 3
    finally:
        monkeypatch.undo()
        cib.tool_entry_points.cache_clear()
        cib.tool_closure.cache_clear()


def test_every_unreached_module_carries_a_disposition(monkeypatch):
    """Nothing sits in the third bucket without a written answer.

    The second half is the discriminator. An `undisposed_modules()` that returned
    `()` unconditionally would satisfy the first assertion forever and would keep
    satisfying it on the day someone adds a module nothing reaches — which is the
    only day the check matters. So a synthetic module is spliced into the tree
    with no disposition written for it, and it has to come back named.
    """
    assert cib.undisposed_modules() == ()

    ghost = f"{cib.PACKAGE}.ghost_module"
    modules = dict(cib.module_table())
    modules[ghost] = cib.SRC / cib.PACKAGE / "ghost_module.py"
    graph = dict(cib.import_graph())
    graph[ghost] = ()
    monkeypatch.setattr(cib, "module_table", lambda: modules)
    monkeypatch.setattr(cib, "import_graph", lambda: graph)
    try:
        assert ghost in cib.unreached_modules()["unreached"], (
            "a module nothing imports must land in the third bucket"
        )
        assert cib.undisposed_modules() == (ghost,), cib.undisposed_modules()
    finally:
        monkeypatch.undo()
        cib.core_closure.cache_clear()
        cib.tool_entry_points.cache_clear()
        cib.tool_closure.cache_clear()
        cib.test_reachers.cache_clear()

    assert cib.undisposed_modules() == ()


def test_a_disposition_naming_no_module_is_a_stale_declaration(monkeypatch):
    """The disposition table rots the way every other declaration in the file does.

    A key naming a module that was renamed or deleted keeps describing it forever,
    and the report keeps printing prose about a file that is not there. Same
    failure as a renamed entry point or an emptied script root, same exit code.
    """
    assert cib.stale_declarations() == []
    monkeypatch.setattr(cib, "UNREACHED_DISPOSITIONS", {
        **cib.UNREACHED_DISPOSITIONS,
        f"{cib.PACKAGE}.module_deleted_three_commits_ago":
            "GONE: nothing in the tree answers to this name any more.",
    })
    try:
        stale = cib.stale_declarations()
        assert any(item["kind"] == "unreached disposition" for item in stale), stale
        assert cib.main([]) == 3
    finally:
        monkeypatch.undo()

    assert cib.stale_declarations() == []


def test_a_disposition_for_a_now_reached_module_is_stale(monkeypatch):
    """A reason nothing reaches a serving-reachable module is the same rot."""
    assert cib.stale_declarations() == []
    reached = f"{cib.PACKAGE}.verdict"
    assert reached in cib.core_closure()
    monkeypatch.setattr(cib, "UNREACHED_DISPOSITIONS", {
        **cib.UNREACHED_DISPOSITIONS,
        reached: "STALE: this module is serving-reachable.",
    })
    try:
        stale = cib.stale_declarations()
        assert {
            "kind": "unreached disposition",
            "name": reached,
        } in stale, stale
        assert cib.main([]) == 3
    finally:
        monkeypatch.undo()

    assert cib.stale_declarations() == []


def test_an_emptied_tests_root_is_a_stale_declaration(monkeypatch, tmp_path):
    """The annotation root must not silently empty either.

    `tests/` moves no module between buckets — it only supplies the sentence that
    says which test covers an unreached one. If the glob stops finding a
    first-party importer, that sentence vanishes and the report goes on exiting 0
    over a claim it can no longer make. Quiet is the failure mode, so it fails loud.
    """
    assert cib.stale_declarations() == []
    monkeypatch.setattr(cib, "TESTS_ENTRY_DIR", tmp_path / "gone")
    cib.test_reachers.cache_clear()
    try:
        stale = cib.stale_declarations()
        assert any(item["kind"] == "tests root" for item in stale), stale
        assert cib.main([]) == 3
    finally:
        monkeypatch.undo()
        cib.test_reachers.cache_clear()

    assert cib.stale_declarations() == []


def test_the_report_states_a_true_sentence_about_the_unreached_bucket(capsys):
    """The printed sentence is the product, and it used to be false.

    "reached by nothing" and "N by nothing at all" were both untrue of every
    module they named: two of the three are imported by tests, one implements an
    interface whose callers are outside this repository, and the guard had
    measured neither before printing. The replacement must state the measure it
    actually ran and then answer per module — with the test-file names DERIVED, so
    renaming tests/test_bind_check.py changes the printed line instead of leaving
    a stale string behind the way a sentence in a docstring would.
    """
    assert cib.main([]) == 0
    out = capsys.readouterr().out

    assert "reached by nothing" not in out, out
    assert "by nothing at all" not in out, out
    assert "provenance or documentation" not in out, out
    parameter_sentence = (
        "None of that covers a path taken as a PARAMETER, which the guard "
        "cannot see at all: src/indra_belief/calibration_constants.py:189 opens "
        "whatever `run_path` its caller hands it. A count of 0 literals in a "
        "call position is not a finding of no path coupling."
    )
    assert parameter_sentence in out, out
    assert f"{cib.PACKAGE}.data.corpus" in out, out
    # The report used to enumerate its BLIND SPOTS, and four rounds of that shipped
    # four incomplete lists — the last named `os.path.join` assembly but not
    # `/`-operator assembly, so `open(ROOT / "data/x.jsonl")` scored `inert` while
    # the sentence implied coverage. A blacklist of blind spots is not completable.
    # So the report now states the RECOGNISED set (finite, checkable) and lets
    # everything else fall outside by construction. Pin that shape, not a word list:
    # the recognised calls must be named from `_FS_CALLS` itself, so adding a reader
    # without disclosing it fails here.
    assert "recognises, exhaustively" in out, out
    for call in sorted(cib._FS_CALLS):
        assert call in out, f"recognised call {call!r} is not disclosed in the report"
    # The load-bearing sentence: what `inert` does and does not license a reader to
    # conclude. Without this, a zero count reads as "no path coupling" — the exact
    # false inference this whole section exists to prevent.
    assert "NEVER 'not a read'" in out, out
    for unmeasured in ("`.joinpath`", "os.path.join", "bound twice", "parameter"):
        assert unmeasured in out, out

    lines = out.splitlines()
    for module in cib.unreached_modules()["unreached"]:
        named = [
            line for line in lines
            if line.startswith("    no closure here reaches:") and module in line
        ]
        assert len(named) == 1, f"{module} should be reported once: {named}"
        line = named[0]
        reason = cib.UNREACHED_DISPOSITIONS[module]
        assert cib._first_sentence(reason) in line, line
        assert len(line) > len(module) + 40, (
            f"a bare name is not a stated reason: {line}"
        )

    reachers = cib.test_reachers()
    assert reachers, "at least one derived test reacher keeps this gate discriminating"
    for module, names in reachers.items():
        assert names, f"a test file does import {module}; the report must say which"
        line = next(line for line in lines if module in line)
        for name in names:
            assert f"{cib.TESTS_ENTRY_DIR.name}/{name}" in line, line


def test_group_by_package_prefix_keeps_a_package_and_its_children_on_one_line():
    """The report branch that no live run exercises, tested directly.

    On the current tree every unreached module is a singleton, so this grouping
    cannot fire in a real run and a reader has no way to tell it is correct.
    The pair it exists for — a package and its child, where the package name is
    a literal PREFIX of the child — is what breaks the substring-based line
    selection in this file's other tests.
    """
    fn = cib.group_by_package_prefix

    # the case it was written for
    assert fn(("indra_belief.probes", "indra_belief.probes.battery")) == [
        ("indra_belief.probes", "indra_belief.probes.battery")
    ]
    # ORDER-INDEPENDENCE. Enumerating the child first must still report each
    # module exactly once. Before the `candidate not in grouped` guard this
    # returned [(battery,), (battery, probes)] — battery on TWO lines, the very
    # duplicate the grouping exists to prevent. It was masked only because
    # `unreached_modules()` returns `tuple(sorted(...))`.
    child_first = fn(("indra_belief.probes.battery", "indra_belief.probes"))
    assert [m for g in child_first for m in g].count("indra_belief.probes.battery") == 1
    assert sorted(m for g in child_first for m in g) == [
        "indra_belief.probes", "indra_belief.probes.battery"
    ]
    # unrelated modules stay separate, and a shared prefix that is NOT a
    # package boundary must not group ("probes" vs "probes_extra")
    assert fn(("a.probes", "a.probes_extra")) == [("a.probes",), ("a.probes_extra",)]
    # today's real shape: all singletons
    assert fn(("a", "b", "c")) == [("a",), ("b",), ("c",)]
