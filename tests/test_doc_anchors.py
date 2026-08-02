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

The section-scoped guard over research/serving_architecture.md has now failed
open three times, each time differently, and each closure is pinned below:

  * whole sections were never opened, because scoping by title covered only the
    titles somebody remembered to register  -> `unregistered_sections`, exit 3;
  * the fixture asserting that contract pinned SECTION_TITLES[0], so the first
    sibling to register a second title turned exit 3 into exit 2 and masked it;
  * the check validated PATHS, not SYMBOLS. Four assembler methods were deleted
    from comparison/replay.py and fourteen citations of them stayed green,
    because `replay.py` still existed  -> `find_dead_symbols`, exit 4.

The last is the one the tests below are new for. What they deliberately do NOT
assert is that every citation is checked: `symbol_coverage` reports a
checked/unchecked split, and the honesty test asserts that the forms this guard
cannot resolve are counted as UNCHECKED rather than silently passing.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_doc_anchors as cda  # noqa: E402
import check_new_section_anchors as cnsa  # noqa: E402


def test_appended_serving_sections_have_zero_dead_anchors():
    """Sections appended to research/serving_architecture.md stay anchor-clean.

    That document cannot join `check_doc_anchors.DOCS` while sections 1-8 carry
    pre-existing unstable numeric anchors, so scripts/check_new_section_anchors.py
    scopes the same `find_dead_anchors` to the appended sections by TITLE. Calling
    it here keeps the guard from rotting into a script nothing runs.
    """
    misses, absent = cnsa.scoped_misses()
    assert absent == [], (
        "appended section heading(s) missing from "
        f"{cnsa.DOC.name}: {', '.join(absent)}"
    )
    assert misses == [], (
        "appended serving-architecture sections contain invalid code anchors:\n"
        + "\n".join(
            f"  {m['doc']}:{m['line']} -> {m['reason']}: {m['path']}"
            for m in misses
        )
    )


def test_serving_doc_has_no_unregistered_sections():
    """Every `## ` section from the first guarded one down is actually checked.

    Title scoping is what lets this document be guarded at all while §1-§8 keep
    their pre-existing debt, but on its own it fails OPEN: an appended section
    whose title nobody registered is not clean, it is unread. This asserts the
    guard's coverage, so `test_appended_serving_sections_have_zero_dead_anchors`
    above is a statement about the whole tail rather than about whichever spans
    happened to be registered.
    """
    unregistered = cnsa.unregistered_sections()
    assert unregistered == [], (
        f"{cnsa.DOC.name} carries section(s) no SECTION_TITLES entry claims, so "
        "their anchors are never checked — register the title(s) in "
        "scripts/check_new_section_anchors.py:\n"
        + "\n".join(f"  line {lineno}: {text}" for lineno, text in unregistered)
    )


def test_unregistered_section_after_a_guarded_one_fails_loudly(
    tmp_path, monkeypatch, capsys
):
    """The seeded sibling that used to pass must now fail, and fail by name.

    Reproduced before the fix: appending an unregistered sibling section citing a
    nonexistent file AND a numeric line anchor left the guard printing OK at
    exit 0 and this module at 3 passed, because scoping by title meant the
    section was never opened. Sibling nodes append to this document in the same
    wave, so fail-open here rots the guard on its first real use.

    Also pinned: sections ABOVE the first guarded one keep their pre-existing
    anchor debt out of scope, and registering the new title turns its formerly
    invisible anchors into reported misses.
    """
    # One section per REGISTERED title, so this fixture stays correct as siblings
    # append entries to SECTION_TITLES — the module's documented extension path.
    # A registered title absent from the document exits 2 (MISSING SECTION),
    # which outranks and would mask the exit-3 contract under test here.
    lines = [
        "# Serving architecture",
        "## 8. Older section carrying pre-existing debt",
        "Legacy numeric anchor, deliberately out of scope: `spend_guard.py:1068-1072`.",
    ]
    for offset, title in enumerate(cnsa.SECTION_TITLES):
        lines.append(f"## {9 + offset}. {title}")
        lines.append(
            "Stable cite: `src/indra_belief/comparison/replay.py` (`main_request`)."
        )
    sibling_lineno = len(lines) + 1
    sibling_heading = f"## {9 + len(cnsa.SECTION_TITLES)}. Sibling section"
    lines += [
        sibling_heading,
        "Dead path `src/indra_belief/totally_bogus_sibling.py`, "
        "numeric `spend_guard.py:1068`.",
    ]
    doc = tmp_path / "serving_architecture.md"
    doc.write_text("\n".join(lines) + "\n")

    assert cnsa.unregistered_sections(doc) == [(sibling_lineno, sibling_heading)], (
        "the unregistered sibling section was not reported — the guard is still "
        "fail-open"
    )

    monkeypatch.setattr(cnsa, "DOC", doc)
    assert cnsa.main() == 3, "an unregistered section must exit non-zero (3)"
    printed = capsys.readouterr().out
    assert "UNREGISTERED SECTION" in printed
    assert sibling_heading in printed

    # Scoping still holds: §8's debt sits above the first guarded heading.
    misses, absent = cnsa.scoped_misses(doc)
    assert (misses, absent) == ([], [])

    # And registering the title is what turns its anchors from unread to read.
    registered = cnsa.SECTION_TITLES + ("Sibling section",)
    assert cnsa.unregistered_sections(doc, registered) == []
    now_seen = {m["path"] for m in cnsa.scoped_misses(doc, registered)[0]}
    assert now_seen == {"src/indra_belief/totally_bogus_sibling.py", "spend_guard.py"}


def test_serving_doc_cites_no_dead_symbols():
    """Every dotted citation this guard can resolve still resolves.

    The regression: K1-prepared-execution deleted `ReplayIndex.main_request`,
    `ReplayIndex._record` and `ScoringRecord.format_user_message`, and the
    fourteen citations of them in the serving doc stayed GREEN — the guard
    asserted that `comparison/replay.py` exists, which it does, and never opened
    it. §9's protocol section was left telling a benchmark operator to call a
    method that is not there.
    """
    dead = cnsa.find_dead_symbols()
    assert dead == [], (
        f"{cnsa.DOC.name} cites symbol(s) the tree no longer defines — rewrite "
        "each to its CURRENT owner, by symbol, never by line:\n"
        + "\n".join(f"  line {m['line']}: {m['symbol']} — {m['reason']}"
                    for m in dead)
    )


def test_a_dead_symbol_exits_four_and_says_where_it_looked(tmp_path, monkeypatch,
                                                           capsys):
    """The seeded proof: the exact citation that shipped green must now fail.

    Reproduced on the live document before this check existed —
    `ReplayIndex.main_request` in §9.1 and `ReplayIndex._record` in the §1
    diagram, guard exit 0, "OK". Both forms are pinned here, the fenced one
    included: four of the fourteen dead citations lived inside the architecture
    diagram, which is a fenced block and therefore invisible to any check that
    reads only inline backtick spans.
    """
    doc = tmp_path / "serving_architecture.md"
    lines = ["# Serving architecture", "```",
             "  ReplayIndex._record()   <- renderer A", "```"]
    for offset, title in enumerate(cnsa.SECTION_TITLES):
        lines.append(f"## {9 + offset}. {title}")
        lines.append("Prompts come from `ReplayIndex.main_request`.")
    doc.write_text("\n".join(lines) + "\n")

    dead = cnsa.find_dead_symbols(doc)
    assert {m["symbol"] for m in dead} == {"ReplayIndex._record",
                                           "ReplayIndex.main_request"}
    # The message must name the file it resolved against, or the maintainer has
    # to guess which of several same-named modules the guard read.
    assert all("comparison/replay.py" in m["reason"] for m in dead)

    monkeypatch.setattr(cnsa, "DOC", doc)
    assert cnsa.main() == 4, "a dead symbol must exit non-zero (4)"
    printed = capsys.readouterr().out
    assert "DEAD SYMBOLS" in printed
    assert "ReplayIndex.main_request" in printed


def test_symbol_check_reports_what_it_cannot_read_as_unchecked(tmp_path):
    """Coverage honesty. A guard that overstates its reach is worse than none.

    Bare names, instance attributes, stdlib modules, ambiguous heads and
    filenames are all UNCHECKED — not clean. `scorer._select_examples` is the
    interesting one: there are two `scorer.py` under src/, so resolving it would
    mean picking one, and picking wrong reports drift that is not there.
    """
    doc = tmp_path / "serving_architecture.md"
    doc.write_text(
        "# Doc\n"
        "Bare name: `parse_response`. Attribute: `self._reservations`.\n"
        "Stdlib: `json.dumps`. Filename: `spend_guard.py`.\n"
        "Ambiguous head: `scorer._select_examples`.\n"
        "Resolvable and real: `ReplayIndex.prepare`, `ExecutionBody.render`.\n"
    )
    assert cnsa.find_dead_symbols(doc) == []
    coverage = cnsa.symbol_coverage(doc)
    assert coverage["missing"] == 0
    assert coverage["checked"] == 2, coverage
    assert coverage["unchecked"] >= 4, coverage

    for chain in ("self._reservations", "json.dumps", "spend_guard.py",
                  "scorer._select_examples"):
        assert cnsa.resolve_symbol(chain)[0] == "unchecked", chain
    for chain in ("ReplayIndex.prepare", "ExecutionBody.render",
                  "comparison.replay", "verdict.parse_response"):
        assert cnsa.resolve_symbol(chain)[0] == "ok", chain
    # A re-export counts: replay.py imports parse_response from verdict.py, so a
    # reader following `replay.parse_response` arrives somewhere real.
    assert cnsa.resolve_symbol("replay.parse_response")[0] == "ok"


def test_coverage_failures_still_outrank_the_symbol_check(tmp_path, monkeypatch):
    """Exit codes 0-3 keep the exact meaning they had before 4 existed.

    A document with BOTH an unregistered section and a dead symbol must still
    report the unregistered section: an unread section is a statement about how
    much of the document was checked at all, and it outranks anything found
    inside the part that was read.
    """
    lines = ["# Serving architecture"]
    for offset, title in enumerate(cnsa.SECTION_TITLES):
        lines.append(f"## {9 + offset}. {title}")
        lines.append("Dead: `ReplayIndex.main_request`.")
    lines += [f"## {9 + len(cnsa.SECTION_TITLES)}. Sibling section", "Body."]
    doc = tmp_path / "serving_architecture.md"
    doc.write_text("\n".join(lines) + "\n")

    assert cnsa.find_dead_symbols(doc), "fixture must carry a dead symbol"
    monkeypatch.setattr(cnsa, "DOC", doc)
    assert cnsa.main() == 3, "unregistered-section coverage must outrank exit 4"


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
