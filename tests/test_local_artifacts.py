"""The loud half of the local-artifact guard. Runs everywhere, skips never.

`tests/_local_artifacts.py` lets a module skip when the gitignored comparison
tree is WHOLLY absent, which is what makes CI green on a fresh checkout. The
failure mode of any skip guard is that it starts skipping on a machine that used
to run, and nothing says so. These tests are the answer to that: they carry no
skip, so they run on CI and on the machine holding the data alike.
"""
from __future__ import annotations

import _local_artifacts as artifacts


def test_the_checkout_is_whole() -> None:
    """A tree with SOME artifacts and not others is a finding, not a clean skip.

    This is the case a skipif can never express, because it is neither "run" nor
    "there was nothing to run". It is a deletion, a half-finished rsync or a
    rename, and it is the exact shape of "a freeze quietly stopped running". One
    failure here, naming what is gone, arrives ahead of the ninety
    `FileNotFoundError`s the dependent modules would otherwise raise.
    """
    assert not artifacts.partly_absent(), (
        "this checkout holds some declared local-artifact trees and not others: "
        f"present={sorted(k for k, v in artifacts.present().items() if v)} "
        f"absent={list(artifacts.missing())}. Fetch the rest or remove them all "
        "— a partial tree runs a subset of the freezes while reporting green."
    )


def test_no_module_shadows_the_guard_with_a_second_pytestmark() -> None:
    """A bare `pytestmark = ...` REPLACES the guard; it does not add to it.

    This is not hypothetical. Two modules already carried their own
    module-level skipif for a different condition, and adding
    `pytestmark = _artifacts.requires()` above them left the guard assigned and
    then immediately overwritten — eight tests kept running on a tree with no
    comparison corpus, and the first verification pass missed it because it
    looked at each module's FIRST pytestmark only.

    The fix is a list, and this is what keeps it a list.
    """
    import ast
    from pathlib import Path

    offenders = []
    for path in sorted(Path(__file__).parent.glob("test_*.py")):
        if path.name == Path(__file__).name:
            continue  # this file is the guard's own test and is never guarded
        source = path.read_text()
        if "import _local_artifacts" not in source:
            continue
        assignments = [
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.Assign)
            and any(getattr(target, "id", "") == "pytestmark" for target in node.targets)
        ]
        if len(assignments) != 1:
            offenders.append(f"{path.name}: {len(assignments)} pytestmark assignments")
            continue
        value = assignments[0].value
        names = {n.attr for n in ast.walk(value) if isinstance(n, ast.Attribute)}
        if "requires" not in names:
            offenders.append(f"{path.name}: pytestmark does not include the guard")
    assert offenders == [], offenders


def test_the_declaration_is_not_empty_and_every_probe_is_inside_the_repo() -> None:
    """The guard cannot be neutered into skipping everything unconditionally.

    An empty probe table would make `wholly_absent()` false and `missing()`
    empty, so every module would run and every module would fail — or, with one
    sign flipped, every module would skip forever. Both are worse than the
    defect this replaces, and neither announces itself.
    """
    assert artifacts.present(), "no local-artifact tree is declared"
    for label, path in artifacts._PROBES.items():
        assert artifacts.ROOT in path.parents, label
