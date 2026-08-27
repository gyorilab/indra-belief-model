"""Refuse to overwrite a gold artifact that is not asked for by name.

WHY THIS EXISTS, measured. The gold builders had no argument parsing at all, so
`python scripts/build_curation_eval.py --help` did not print help — it rebuilt
`eval_curation_v1.jsonl` and its sidecar in place. `--help` is the safest thing
anyone types at an unfamiliar script, and it silently replaced a benchmark that
digest tests pin.

WHAT REBUILDING MEANS, and what it does not. Re-running a builder is legitimate
and the method is what reproduces, not the bytes: these scripts derive their
sample by EXCLUDING every pair that appears in the prior holdouts, eval sets and
fewshot pools present on disk. More of those exist now than when any given gold
was first drawn, so the same seed correctly yields a smaller fresh pool — that
is the leakage rule working, not drift. What must not happen is a rebuilt
artifact quietly taking the place of the one a published number was measured on.
So a rebuild is a deliberate act with a new artifact at the end of it, and the
digest pins in tests/ move with it.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def add_rebuild_flag(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="redraw the sample and REPLACE the existing artifact. Without this "
             "an existing output is left alone and the run refuses.",
    )
    return parser


def guard_outputs(paths: list[Path] | tuple[Path, ...], *, rebuild: bool) -> None:
    """Raise SystemExit unless every existing output was explicitly asked for.

    Called at the TOP of main(), before any sampling work, so a refusal costs
    nothing and a `--help` never reaches the builder at all.
    """
    existing = [p for p in paths if Path(p).exists()]
    if not existing or rebuild:
        return
    listed = "\n".join(f"    {p}" for p in existing)
    raise SystemExit(
        "refusing to overwrite an existing gold artifact:\n"
        f"{listed}\n\n"
        "These are pinned by digest in tests/ and are what published numbers "
        "were measured on. Re-running is legitimate — the METHOD reproduces, "
        "and a smaller fresh pool is the leakage rule correctly excluding gold "
        "that has been drawn since. But the result is a NEW artifact, so:\n"
        "  1. pass --rebuild to redraw and replace, then\n"
        "  2. update the digest pins in tests/, and\n"
        "  3. do not compare new numbers against ones measured on the old draw.\n"
    )
