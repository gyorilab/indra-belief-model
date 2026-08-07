"""Which tests need the gitignored local artifacts, and what happens when they are absent.

THE PROBLEM THIS SOLVES, measured rather than described. On a fresh checkout —
`git worktree add` gives one, since a worktree carries only tracked files — the
suite reports **86 failed, 6 errors, 1422 passed** across eighteen test files.
CI runs `python -m pytest -q` as its second step, so it has been RED on every
push for at least eight consecutive commits, and every step after it (the
contamination guard, the doc-anchor guard, the viewer checks, the deck build)
was skipped and has not run either. `research/kernel_unification_findings.md`
§7.2 item 6 described this as one test.

Not one of those failures is a defect in the code under test. Each is a test
opening a path that only exists on a machine holding the ~20 GB of gitignored
comparison tree: the runs and their ledgers, the published bundles, the frozen
substrate, and the paper-comparison artifacts under `data/results/` that are not
whitelisted in `.gitignore`.

WHY NOT SIMPLY SKIP. Item 6 states the trade honestly and both halves are real:
a skip guard lets a freeze silently stop running, and no guard makes CI fail on
a fresh checkout. Neither "always skip" nor "always crash" is acceptable, so the
question is three-valued and each answer gets a different treatment.

  ALL PRESENT      run. A difference is a FAILURE and never a skip. This is the
                   local machine, which is the only place these can run at all.
  WHOLLY ABSENT    skip, naming what is needed. This is CI and any fresh
                   checkout. Nothing is hidden: there was never any data to
                   check against.
  PARTLY ABSENT    do not skip. `tests/test_local_artifacts.py` fails, once,
                   with the list — a tree where some artifacts resolve and some
                   do not is a deletion, a half-finished rsync or a rename, and
                   every one of those is a finding. The dependent tests then
                   fail on their own too, which is correct; what this buys is
                   ONE failure that says what happened, ahead of ninety that
                   say `FileNotFoundError`.

The residual exposure — a machine that HAS the data and stops running a freeze
because someone deleted it — is exactly the PARTLY ABSENT case, which is why it
is a failure rather than a skip.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# The gitignored trees these tests read. Declared as ROOTS rather than as a list
# of individual files on purpose: a file list has to be maintained against every
# new artifact and rots into the same hand-maintained-list defect that
# `scripts/check_doc_anchors.py`'s DOCS glob and
# `scripts/check_import_boundary.py`'s core derivation were both rewritten to
# avoid. A root is a claim about a whole tree, and a tree is either fetched or
# it is not.
#
#   data/comparison    every run directory, spend ledger, published bundle and
#                      frozen replay substrate. ~20 GB. `.gitignore:102`.
#   data/results       MIXED — several artifacts here are whitelisted per file
#                      and tracked (the /paper figures). The untracked ones are
#                      the paper-comparison golds and the `current_indra_*`
#                      re-derivations. Presence is therefore judged on the
#                      specific untracked members below, never on the directory.
#   data/corpora       generated corpus inputs. `.gitignore` keeps all but two
#                      tracked fixtures.
_COMPARISON = ROOT / "data" / "comparison"
_RESULTS = ROOT / "data" / "results"
_CORPORA = ROOT / "data" / "corpora"

# One untracked member per tree, used only as the presence probe. These are not
# the whole dependency — they are the cheapest thing that is present exactly
# when the tree was fetched.
_PROBES: dict[str, Path] = {
    "data/comparison": _COMPARISON / "models",
    "data/results (untracked paper-comparison artifacts)": (
        _RESULTS / "indra_belief_comparison_metrics.json"
    ),
    "data/corpora (generated corpus inputs)": _CORPORA / "cogex_evidence_sample.jsonl",
}

_INSTRUCTION = (
    "these are gitignored local artifacts; run on the machine holding them with "
    "PYTHONPATH=src .venv/bin/python -m pytest"
)


def present() -> dict[str, bool]:
    """Which declared trees this checkout holds. The only filesystem read here."""
    return {label: path.exists() for label, path in _PROBES.items()}


def missing() -> tuple[str, ...]:
    """The declared trees this checkout does NOT hold, in declaration order."""
    return tuple(label for label, exists in present().items() if not exists)


def wholly_absent() -> bool:
    """True when NONE of the declared trees are here — a fresh checkout or CI."""
    return not any(present().values())


def partly_absent() -> bool:
    """True when some trees are here and some are not. Never a clean skip."""
    states = set(present().values())
    return states == {True, False}


def requires(*, trees: tuple[str, ...] = ()) -> pytest.MarkDecorator:
    """Module-level mark: skip only when the local artifacts are WHOLLY absent.

    `trees` narrows the claim to specific declared roots for a module that needs
    only some of them; the default is "any of them", which is right for a module
    that reads across the comparison tree. It never skips on a PARTIAL checkout
    — see this module's docstring — so a deletion surfaces as a failure rather
    than as a green run over less than it claims.
    """
    absent = missing() if not trees else tuple(t for t in trees if t in missing())
    skip = wholly_absent() if not trees else len(absent) == len(trees)
    return pytest.mark.skipif(
        skip,
        reason=f"absent: {', '.join(absent) or 'local artifacts'} — {_INSTRUCTION}",
    )
