"""Guard the doc-anchor guard (node R6).

scripts/check_doc_anchors.py fails CI when a live task-hypergraph doc cites a
missing code path or a volatile numeric line coordinate, including basename-only
coordinates — the anchor-drift class that let the removed `composed_scorer.py`
linger in the calibration doc after the config-scoped hybrid ship. These tests
make that class of regression fail:

  * the current live docs contain ZERO dead anchors (post-R2/R3 repair), and
  * a synthetic doc citing the removed composed_scorer.py IS flagged — proving
    the guard would have caught finding #6 — while an anchor sitting on a line
    explicitly marked "(superseded)" is correctly skipped.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_doc_anchors as cda  # noqa: E402


def test_live_docs_have_zero_dead_anchors():
    """Every explicit source path resolves and every live cite is symbol-based.

    A non-empty result is real anchor drift: a moved/deleted file or numeric
    coordinate cited outside a superseded/historical or planned context.
    """
    misses = cda.find_dead_anchors()
    assert misses == [], (
        "live task-hypergraph docs contain invalid code anchors:\n"
        + "\n".join(
            f"  {m['doc']}:{m['line']} -> {m['reason']}: {m['path']}"
            for m in misses
        )
    )


def test_synthetic_dead_and_numeric_anchors_flagged_and_superseded_skipped(tmp_path):
    """A synthetic doc proves the guard's load-bearing behaviours:
      * a cite of the removed src/indra_belief/composed_scorer.py IS flagged
        (this is finding #6 — the anchor drift the guard exists to catch);
      * full-path, basename, and suffix numeric coordinates are flagged; and
      * superseded lines, URLs, versions, and non-source prose are not flagged.
    """
    doc = tmp_path / "synthetic_hypergraph.md"
    doc.write_text(
        "# Synthetic hypergraph\n"
        "The scorer lives in `src/indra_belief/composed_scorer.py:12` today.\n"
        "Volatile live cite: `scripts/check_doc_anchors.py:99`.\n"
        "Volatile basename cite: `noise_model.py:52-76`.\n"
        "Volatile suffix cite: `scorers/monolithic/scorer.py:59`.\n"
        "Stable viewer cite: `viewer/src/lib/data/queries.ts` (`getRunCalibration`).\n"
        "A URL is not a repo cite: `https://example.org/src/url_only.py:12`.\n"
        "Versions and prose are not cites: Python 3.12; `runtime.py:3.12`; `schema_version: 8`.\n"
        "Old numeric note: `noise_model.py:12` (historical) — ignore.\n"
        "Old note: `src/indra_belief/gone_but_marked.py` (superseded) — ignore.\n"
    )
    misses = cda.find_dead_anchors([doc])
    flagged = {m["path"] for m in misses}

    # Finding #6: the removed composed_scorer.py must be caught.
    assert "src/indra_belief/composed_scorer.py" in flagged, (
        "guard failed to flag the removed composed_scorer.py — it would NOT "
        "have caught finding #6"
    )
    # The (superseded)-marked missing path must be skipped, not flagged.
    assert "src/indra_belief/gone_but_marked.py" not in flagged, (
        "guard flagged an anchor on an explicitly (superseded) line — the "
        "skip is not working"
    )
    numeric = [m for m in misses if m["path"] == "scripts/check_doc_anchors.py"]
    assert numeric and "unstable numeric anchor" in numeric[0]["reason"]
    assert any(m["path"] == "noise_model.py" for m in misses)
    assert any(m["path"] == "scorers/monolithic/scorer.py" for m in misses)
    assert "viewer/src/lib/data/queries.ts" not in flagged
    assert "src/url_only.py" not in flagged
    assert "runtime.py" not in flagged
