"""Guard the doc-anchor guard.

`scripts/check_doc_anchors.py` fails CI when a live research doc cites a missing
code path, a volatile numeric line coordinate (basename-only coordinates
included), or a dotted symbol the tree no longer defines — the anchor-drift class
that let the removed `composed_scorer.py` linger in the calibration doc after the
config-scoped hybrid ship. These tests make that class of regression fail.

What is pinned here, and why each pin exists:

  * THE CORPUS SCAN. Every `research/*.md` carries zero dead anchors and zero
    dead symbols, at a grandfather allowance of zero. `DOCS` is a glob, so both
    assertions are order-independent with respect to sibling nodes: a document
    that lands mid-wave is scanned by whichever run happens after it, and there
    is no list anybody has to remember to update.
  * THE SHRINK-ONLY RATCHET. `GRANDFATHERED_ANCHORS` is the one place a document
    may declare pre-existing numeric debt, and it may only count DOWN. Three
    tests cover its three transitions: debt that shrank hands the maintainer the
    new number, a key with nothing left tells them to delete it, and an
    occurrence beyond the allowance comes back as an ordinary invalid anchor
    rather than as a table complaint. That last one is what stops a repair in one
    part of a file being silently traded for a new anchor in another part of it.
    All three run against a `tmp_path` document and a `monkeypatch.setitem`, so
    none of them asserts anything about the live table's contents.
  * THE EXIT PRECEDENCE 1 -> 3 -> 4, pinned across three states of one seeded
    document. The three codes name repairs of different kinds and collapsing them
    would destroy the diagnosis.
  * COVERAGE HONESTY. What these tests deliberately do NOT assert is that every
    citation is checked: `symbol_coverage` reports a checked/unchecked split, and
    the honesty test asserts that the forms this guard cannot resolve are counted
    as UNCHECKED rather than silently passing.
  * THE ANCHOR GRAMMAR'S RIGHT BOUNDARY, which is what keeps the extension
    alternation from biting into a longer word and inventing a citation nobody
    wrote.
  * THE RETIREMENT of the section-scoped shim, so its deletion is durable rather
    than a diff somebody re-adds.

The symbol check is the youngest of these and the reason is recorded in
`test_a_dead_symbol_exits_four_and_says_where_it_looked`: four assembler methods
were deleted from `comparison/replay.py` and fourteen citations of them stayed
green, because the guard asserted that `replay.py` exists — which it does — and
never opened it.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_doc_anchors as cda  # noqa: E402


def test_live_docs_cite_no_dead_symbols():
    """Every dotted citation this guard can resolve still resolves, corpus-wide.

    The regression: K1-prepared-execution deleted `ReplayIndex.main_request`,
    `ReplayIndex._record` and `ScoringRecord.format_user_message`, and the
    fourteen citations of them in the serving doc stayed GREEN — the guard
    asserted that `comparison/replay.py` exists, which it does, and never opened
    it. §9's protocol section was left telling a benchmark operator to call a
    method that is not there.

    Scoped to the whole corpus rather than to that one document, because the
    first time this same check was run over every `research/*.md` it reported
    four further dead symbols in two OTHER documents, one of them a live
    operational runbook. A pass here is a statement about the corpus; a pass
    scoped to one file was read as one and was not.
    """
    dead = cda.find_all_dead_symbols()
    assert dead == [], (
        "live research docs cite symbol(s) the tree no longer defines — rewrite "
        "each to its CURRENT owner, by symbol, never by line:\n"
        + "\n".join(f"  {m['doc']}:{m['line']} -> {m['symbol']} — {m['reason']}"
                    for m in dead)
    )


def test_a_dead_symbol_exits_four_and_says_where_it_looked(tmp_path, capsys):
    """The seeded proof: the exact citation that shipped green must now fail.

    Reproduced on the live document before this check existed —
    `ReplayIndex.main_request` in §9.1 and `ReplayIndex._record` in the §1
    diagram, guard exit 0, "OK". Both forms are pinned here, the fenced one
    included: four of the fourteen dead citations lived inside the architecture
    diagram, which is a fenced block and therefore invisible to any check that
    reads only inline backtick spans.
    """
    doc = tmp_path / "serving_architecture.md"
    doc.write_text("# Serving architecture\n"
                   "```\n"
                   "  ReplayIndex._record()   <- renderer A\n"
                   "```\n"
                   "Prompts come from `ReplayIndex.main_request`.\n")

    dead = cda.find_dead_symbols(doc)
    assert {m["symbol"] for m in dead} == {"ReplayIndex._record",
                                           "ReplayIndex.main_request"}
    # The message must name the file it resolved against, or the maintainer has
    # to guess which of several same-named modules the guard read.
    assert all("comparison/replay.py" in m["reason"] for m in dead)

    assert cda.main([str(doc)]) == 4, "a dead symbol must exit non-zero (4)"
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
    assert cda.find_dead_symbols(doc) == []
    coverage = cda.symbol_coverage(doc)
    assert coverage["missing"] == 0
    assert coverage["checked"] == 2, coverage
    assert coverage["unchecked"] >= 4, coverage

    for chain in ("self._reservations", "json.dumps", "spend_guard.py",
                  "scorer._select_examples"):
        assert cda.resolve_symbol(chain)[0] == "unchecked", chain
    for chain in ("ReplayIndex.prepare", "ExecutionBody.render",
                  "comparison.replay", "verdict.parse_response"):
        assert cda.resolve_symbol(chain)[0] == "ok", chain
    # A re-export counts: replay.py imports parse_response from verdict.py, so a
    # reader following `replay.parse_response` arrives somewhere real.
    assert cda.resolve_symbol("replay.parse_response")[0] == "ok"


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


def test_stale_grandfather_entry_that_shrank_hands_over_the_new_number(
    tmp_path, monkeypatch
):
    """Debt that shrank is reported WITH the number to paste in.

    The table is shrink-only, which is only true if a repair is FORCED back into
    it: an allowance nobody spends is a standing licence, and the next regression
    in that file lands inside it silently. Asserting a non-zero exit alone would
    not be enough to make that work — the value of this path is that the
    maintainer is handed `set it to 1` instead of being left to re-count by hand,
    which is exactly how a table drifts away from the document it describes.
    """
    doc = tmp_path / "grandfathered.md"
    doc.write_text("# Doc\nVolatile basename cite: `noise_model.py:285`.\n")
    monkeypatch.setitem(cda.GRANDFATHERED_ANCHORS, cda._rel(doc),
                        {"noise_model.py:285": 2})

    records = cda.find_dead_anchors([doc])
    assert len(records) == 1, records
    reason = records[0]["reason"]
    assert reason.startswith("stale grandfather"), reason
    assert "allowed 2, found 1" in reason, reason
    assert "set it to 1" in reason, reason
    assert cda.main([str(doc)]) == 3, "a stale table entry must exit 3"


def test_grandfather_key_with_no_occurrences_left_says_delete_the_key(
    tmp_path, monkeypatch
):
    """A fully repaired key must say so rather than sit on unspent slack.

    Zero remaining occurrences is the one case where the obvious behaviour — stay
    quiet, the document is clean — is the wrong one. The key survives as an
    allowance for anchors nobody has written yet, so the guard fails and names the
    repair as a DELETION, distinguishing it from the shrink case above.
    """
    doc = tmp_path / "repaid.md"
    doc.write_text("# Doc\nCite by symbol: `src/indra_belief/verdict.py` "
                   "(`parse_response`).\n")
    monkeypatch.setitem(cda.GRANDFATHERED_ANCHORS, cda._rel(doc),
                        {"noise_model.py:285": 3})

    records = cda.find_dead_anchors([doc])
    assert len(records) == 1, records
    reason = records[0]["reason"]
    assert "found 0" in reason, reason
    assert "(delete the key)" in reason, reason
    assert cda.main([str(doc)]) == 3, "an unspent key must exit 3"


def test_occurrences_beyond_the_allowance_are_ordinary_invalid_anchors(tmp_path):
    """Debt above the allowance is a document defect, not a table complaint.

    At the default allowance — zero, where every live document sits — a numeric
    anchor must be reported exactly as it would be with no table in the picture
    at all: an ordinary record naming the cited path and the coordinate, exit 1,
    with none of the stale-table wording. This is the half of the ratchet that
    stops a repair being silently traded for a new anchor elsewhere in the same
    file; without it the two would cancel and the count would still balance.
    """
    doc = tmp_path / "excess.md"
    doc.write_text("# Doc\nVolatile basename cite: `noise_model.py:285`.\n")

    records = cda.find_dead_anchors([doc])
    assert len(records) == 1, records
    assert records[0]["path"] == "noise_model.py", records
    assert "unstable numeric anchor :285" in records[0]["reason"], records
    assert not records[0]["reason"].startswith("stale grandfather"), records
    assert cda.main([str(doc)]) == 1, "an over-allowance anchor must exit 1"


def test_exit_precedence_is_invalid_anchor_then_stale_table_then_dead_symbol(
    tmp_path, monkeypatch
):
    """1 -> 3 -> 4, pinned across three states of one seeded document.

    The order is not arbitrary and the codes must not be collapsed, because each
    names a repair of a different KIND. 1 is a broken DOCUMENT — a cited file is
    missing, or a coordinate has to be deleted — and it comes first because it is
    the only one where the prose itself is wrong. 3 is a correct document with an
    out-of-date ALLOWANCE: nothing in the text needs touching, the repair is a
    number in the guard's own table, and the guard prints it. 4 is a document
    whose paths all resolve but whose SYMBOLS do not, repaired by finding out
    where the logic went and renaming the citation to its current owner — the
    slowest of the three, and the one worth surfacing alone once the other two
    are clear.
    """
    doc = tmp_path / "precedence.md"
    key = cda._rel(doc)
    monkeypatch.setitem(cda.GRANDFATHERED_ANCHORS, key, {"noise_model.py:285": 1})

    doc.write_text("# Doc\n"
                   "Missing path: `src/indra_belief/totally_bogus.py`.\n"
                   "Dead symbol: `ReplayIndex.main_request`.\n")
    assert cda.main([str(doc)]) == 1, "an invalid anchor outranks the other two"

    doc.write_text("# Doc\nDead symbol: `ReplayIndex.main_request`.\n")
    assert cda.main([str(doc)]) == 3, "a stale table entry outranks a dead symbol"

    monkeypatch.delitem(cda.GRANDFATHERED_ANCHORS, key)
    assert cda.main([str(doc)]) == 4, "the dead symbol is what remains"


def test_anchor_grammar_stops_at_the_extension_boundary():
    """The right boundary on both anchor patterns, and the live defect it repairs.

    Without a trailing `(?![A-Za-z0-9])` the extension alternation bites into a
    longer word and invents a citation nobody wrote: `src/indra_belief/scorer.python`
    matched as `src/indra_belief/scorer.py`, and the guard then reported a missing
    file against a path that appears in no document. That false positive is a
    `.py` one with no `sh` anywhere in it, which is why the boundary belongs on
    the alternation rather than on whichever extension was added last.
    `hashlib.sha256`, `plan.sha256`, `x.shuffle` and `scripts/foo.shuffle` are the
    same defect read through the `.sh` arm.

    The `.sh` arm is not dead grammar — the corpus really does cite shell scripts,
    and the breakdown below is derived by running `cda._ANCHOR` over `cda.DOCS`
    rather than by grepping for `.sh`. A hand grep gives a different and wrong
    figure, because it also sees bare basenames and a crontab absolute path that
    this pattern's left lookbehind rejects by design.
    """
    assert [m.group("path") for m in
            cda._ANCHOR.finditer("scripts/supervise_comparison_all.sh")] == [
        "scripts/supervise_comparison_all.sh"]

    for text in ("hashlib.sha256", "plan.sha256", "x.shuffle",
                 "scripts/foo.shuffle", "src/indra_belief/scorer.python"):
        assert not cda._ANCHOR.findall(text), f"_ANCHOR matched {text!r}"
        assert not cda._NUMERIC_SOURCE_ANCHOR.findall(text), (
            f"_NUMERIC_SOURCE_ANCHOR matched {text!r}")

    cited = Counter(
        m.group("path")
        for doc in cda.DOCS
        for line in doc.read_text().splitlines()
        for m in cda._ANCHOR.finditer(line)
        if m.group("path").endswith(".sh")
    )
    assert cited == {
        "scripts/supervise_comparison_arm.sh": 1,
        "scripts/supervise_comparison_all.sh": 2,
        "scripts/monitor_comparison_fleet.sh": 1,
        "scripts/serve_mlx.sh": 1,
    }, (
        "the corpus's matchable `.sh` citations moved. This pin exists to keep the "
        "`.sh` arm from becoming grammar that matches nothing, so the repair is to "
        "RE-DERIVE with cda._ANCHOR over cda.DOCS and paste the new breakdown in — "
        f"never to hand-grep for '.sh'. Measured now: {dict(cited)}"
    )


def test_section_scoped_shim_stays_retired():
    """The retired section-scoped guard must not come back.

    It scoped the anchor scan to a hand-registered list of section titles in
    `research/serving_architecture.md`, because §1-§8 of that document carried
    numeric debt that had to stay quarantined while new sections were appended to
    it. That debt reached zero, the document joined the corpus-wide scan at an
    allowance of zero, and the shim became a second guard that could only ever
    agree with the first — while still failing open on any section nobody
    remembered to register. Its one durable part, the symbol check, moved into
    `scripts/check_doc_anchors.py` and is pinned by the tests above.
    """
    shim = ROOT / "scripts" / "check_new_section_anchors.py"
    assert not shim.exists(), (
        f"{shim.relative_to(ROOT)} is back. The section-scoped guard was retired "
        "once research/serving_architecture.md's numeric debt reached zero: the "
        "whole corpus is scanned unscoped by scripts/check_doc_anchors.py, so a "
        "title registry adds no coverage and restores a fail-open path — a "
        "section nobody registers is UNREAD, not clean. A document that genuinely "
        "needs an allowance declares it in GRANDFATHERED_ANCHORS, which may only "
        "shrink."
    )
