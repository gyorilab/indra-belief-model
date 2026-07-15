"""Guard the doc-anchor guard (node R6).

scripts/check_doc_anchors.py fails CI when a live task-hypergraph doc cites a
`src/`/`scripts/` path that no longer exists on disk — the anchor-drift class
that let the removed `composed_scorer.py` linger in the calibration doc after
the config-scoped hybrid ship. These tests make that class of regression fail:

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
    """Every src/scripts path cited by the live hypergraph docs resolves on
    disk. A non-empty result is real anchor drift (a moved/deleted file still
    cited outside a superseded/historical or explicitly-planned context)."""
    misses = cda.find_dead_anchors()
    assert misses == [], (
        "live task-hypergraph docs cite src/scripts paths that don't exist:\n"
        + "\n".join(f"  {m['doc']}:{m['line']} -> {m['path']}" for m in misses)
    )


def test_synthetic_dead_anchor_flagged_and_superseded_skipped(tmp_path):
    """A synthetic doc proves the guard's two load-bearing behaviours:
      * a cite of the removed src/indra_belief/composed_scorer.py IS flagged
        (this is finding #6 — the anchor drift the guard exists to catch);
      * a missing path on a line marked "(superseded)" is NOT flagged.
    """
    doc = tmp_path / "synthetic_hypergraph.md"
    doc.write_text(
        "# Synthetic hypergraph\n"
        "The scorer lives in `src/indra_belief/composed_scorer.py:12` today.\n"
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
