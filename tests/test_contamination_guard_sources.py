"""Guard the contamination-guard's own sources against its own sources.

scripts/check_contamination.py scans every fewshot source the model sees
during inference. A wrong module path once let Source 1
(CONTRASTIVE_EXAMPLES) silently load ZERO examples because the
ModuleNotFoundError was swallowed by a blanket ``except``. These tests
make that class of regression fail in CI:

  * Source 1 (CONTRASTIVE_EXAMPLES) must load >= 1 example.
  * Every declared source either contributes >= 1 example OR is
    legitimately empty — but a broken IMPORT must raise loudly
    (SourceImportError), never silently contribute zero.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_contamination as cc  # noqa: E402


def _counts_by_source(examples: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for ex in examples:
        out[ex["source"]] = out.get(ex["source"], 0) + 1
    return out


def test_source1_contrastive_examples_loads_nonempty():
    """Source 1 — the legacy monolithic CONTRASTIVE_EXAMPLES — must load
    at least one example. A zero here is the exact silent-failure the
    swallowed ModuleNotFoundError used to produce."""
    examples = cc._load_legacy_examples()
    counts = _counts_by_source(examples)
    assert counts.get("CONTRASTIVE_EXAMPLES", 0) >= 1, (
        "Source 1 (CONTRASTIVE_EXAMPLES) loaded zero examples — the "
        "contamination scan is blind to the monolithic fewshots. Check the "
        "import path in _load_legacy_examples."
    )


def test_contrastive_examples_import_is_loud_not_swallowed(monkeypatch):
    """A broken Source-1 import must raise SourceImportError, NOT silently
    yield an empty list. We simulate the broken-module-path scenario by
    making the underlying import fail."""
    import builtins

    real_import = builtins.__import__

    def _boom(name, *args, **kwargs):
        if name == "indra_belief.scorers.monolithic._prompts":
            raise ImportError("simulated broken module path")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _boom)
    with pytest.raises(cc.SourceImportError):
        cc._load_legacy_examples()


def test_every_declared_source_loads_nonempty_or_fails_loud():
    """End-to-end: union of all declared fewshot sources. Every source key
    that appears must carry >= 1 example (an empty source simply would not
    appear). If any declared source's import were broken, load_all_examples
    would raise SourceImportError rather than reach this assertion — so a
    clean return here proves no source silently zeroed out.

    We additionally assert the union is non-trivial and that the historically
    fragile Source 1 is present, guarding against a future refactor that
    drops it without noticing.
    """
    examples = cc.load_all_examples()
    assert examples, "load_all_examples returned nothing — all sources empty?"
    counts = _counts_by_source(examples)
    # Every reported source has a positive count by construction; assert it
    # explicitly so the contract is enforced, not assumed.
    for source, n in counts.items():
        assert n >= 1, f"declared source {source!r} reported a non-positive count"
    # Source 1 must be in the union.
    assert counts.get("CONTRASTIVE_EXAMPLES", 0) >= 1


def test_representative_curation_gold_is_in_default_eval_guard():
    paths = cc._default_eval_paths(str(ROOT / "data/benchmark/holdout_large.jsonl"))
    assert ROOT / "data/benchmark/representative_indra_curations_400.jsonl" in paths
