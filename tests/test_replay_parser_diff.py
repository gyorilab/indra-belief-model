"""The frozen population behind `scripts/replay_parser_diff.py`.

`indra_belief.verdict` replaced three verdict parsers with one, and the evidence
that the replacement reads the corpus the same way is that script: it re-reads
every stored LLM response in `data/comparison*` with all four parsers, then
manufactures the population the stored corpus cannot contain (truncation
mutants) and re-reads those too. Nothing ran it —
`research/kernel_unification_findings.md` §7.2 item 3 records exactly that gap —
so the numbers `src/indra_belief/verdict.py`, `tests/test_verdict_parser.py` and
that document all cite lived only in a terminal scrollback.

This module is their home. It freezes what the script measured, and re-measures
everything that can be re-measured WITHOUT the corpus.

WHAT RUNS ON A FRESH CHECKOUT. Three groups need no path under `data/`, because
the fixture carries the response TEXTS themselves, untruncated:

  (a) the six truncation mutants a parser pair disagreed on. Every one is re-read
      by all four parsers here, so the divergence is a live measurement rather
      than a recorded string, and the property behind it — the batch parser lost
      a verdict the live parser and the new one both recover — is asserted on
      each. So is the phrase that discriminates them.
  (b) 29 stored responses all four parsers agree on. Bit-rot detection, NOT
      population evidence: see the group's own docstring for what 29 texts
      chosen first-two-per-log can and cannot show.
  (c) the seeded constants that define the mutant population, and the three
      prose files that cite the measurement.

WHAT NEEDS THE CORPUS. Group (d) — the population itself. The 15 attempt logs
run to 8.5 GiB and are a gitignored published artifact (`.gitignore:102,113,123`),
so the 238,039-row scan cannot run on `ubuntu-latest` and is SKIPPED there. It
is skipped only when the corpus is WHOLLY ABSENT; when it is present and the
numbers differ, it FAILS. There is no env var that disables it, and the fixture
is never edited to match a new measurement — a diff there is a corpus change or
a parser change, and reconciling it (including the prose citations group (c)
guards) is a human's job.

READ-ONLY, like the script. Every handle is opened "r". Because all three corpus
directories are gitignored, `git status` could never observe a mutation, so the
positive proof is a per-log `(size, st_mtime_ns)` recorded at import and
re-asserted after the scan. The sibling `tests/test_prepared_execution_goldens.py`
sha256s its substrate; at 8.5 GiB that is not affordable and `stat` is free.

Fixture shape follows that same sibling: sorted JSON with `_readme` /
`_regenerate` provenance keys, and one regeneration path that is the only way
the file is ever produced.

Regeneration (never automatic — a diff here is a measurement change):

    PARSER_DIFF_REGEN=1 PYTHONPATH=src .venv/bin/python -m pytest -q \
        tests/test_replay_parser_diff.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.replay_parser_diff import (  # noqa: E402
    CUTS,
    LLM_TIERS,
    PAIRS,
    READERS,
    RESERVOIR,
    SEED,
    Diff,
    _attempt_files,
    run,
)

_ROOT = Path(__file__).resolve().parents[1]
_GOLDENS = Path(__file__).resolve().parent / "goldens" / "parser_diff_population.json"

_README = (
    "X2-parser-diff — the frozen population of scripts/replay_parser_diff.py. "
    "Captured 2026-08-03 at commit ccf53bf against the 15 gitignored attempt "
    "logs under data/comparison*/runs/*/attempts.jsonl (8.5 GiB). Frozen here: "
    "the Part A / Part B totals and row census, the seeded constants that define "
    "the mutant population, the full untruncated text of every mutant a parser "
    "pair disagreed on, and a small sample of stored responses all four parsers "
    "agree on. A diff here is a corpus or parser change, not a test to update."
)
_REGENERATE = (
    "PARSER_DIFF_REGEN=1 PYTHONPATH=src .venv/bin/python -m pytest -q "
    "tests/test_replay_parser_diff.py"
)

# Measured before this module existed, by running the script itself and reading
# its report. Repeated OUTSIDE the fixture so a regeneration cannot quietly
# re-baseline them, exactly as tests/test_prepared_execution_goldens.py::_MEASURED
# does. These are the numbers the three prose files below cite.
_MEASURED = {
    "files": 15,
    "stored_rows": 238_039,
    "stored_considered": 228_812,
    "mutant_rows": 16_756,
    "mutant_considered": 16_756,
    "old_live_ne_old_batch": 6,
    "old_batch_ne_new": 6,
    "off_grid": 11_645,
}

# Which files cite which measurement, by the literal formatted substring — the
# form that survives any rewording of the sentence around it.
_CITES_CONSIDERED = (
    "src/indra_belief/verdict.py",
    "tests/test_verdict_parser.py",
    "research/kernel_unification_findings.md",
)
_CITES_OFF_GRID = ("research/kernel_unification_findings.md",)

# The phrase that separates the live reading from the batch one on every
# divergent mutant. The batch verdict patterns (replay.py, copied verbatim into
# the script) accept only "is" / "=" after the verdict word, so a reply that
# wrote "Verdict should be incorrect" committed for the live parser and for the
# new one, and read as ABSENT for the batch parser — which is the whole
# divergence. Asserted as a measured property of all six texts, not as prose.
_MODAL_VERDICT_PHRASE = re.compile(
    r'(?i)\b(?:verdict|decision|answer)\s+(?:should be|would be)\s*[:"\'*]*\s*'
    r"(correct|incorrect)"
)

_LIVE_READER = dict(READERS)["old_live"]

# Taken at import, before this module opens a single corpus byte.
_CORPUS = _attempt_files()


def _log_stats() -> dict[str, tuple[int, int]]:
    """`(size, st_mtime_ns)` per attempt log — the affordable tamper check.

    sha256 of 8.5 GiB is not affordable in a test; `stat` is free. And a git
    check is impossible: .gitignore:102 (data/comparison/), :113
    (data/comparison_noreason/*) and :123 (data/comparison_verdict_only/*) mean
    `git status` can never observe a mutation to any of these files.
    """
    return {
        str(path.relative_to(_ROOT)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in _attempt_files()
    }


_STATS_AT_IMPORT = _log_stats()

requires_corpus = pytest.mark.skipif(
    not _CORPUS,
    reason=(
        "data/comparison*/runs/*/attempts.jsonl absent (gitignored published "
        "artifact); run locally with PYTHONPATH=src .venv/bin/python -m pytest "
        "tests/test_replay_parser_diff.py"
    ),
)


# --------------------------------------------------------------------------
# Regeneration — the ONLY way tests/goldens/parser_diff_population.json is made
# --------------------------------------------------------------------------

def _divergent_mutants(result: dict) -> list[dict]:
    """Every Part B mutant any pair disagreed on, deduplicated BY LABEL.

    `Diff.examples` is keyed by `(pair, left reading, right reading)`, and each
    divergence is recorded under both the `old_live!=old_batch` key and the
    `old_batch!=new` one — a naive concatenation would yield each mutant twice.
    The caller passes `examples_cap=10**9` so these buckets hold EVERY
    divergence rather than the five per class the report prints.
    """
    texts: dict[str, str] = {}
    for bucket in result["mutants"].examples.values():
        for label, text in bucket:
            texts.setdefault(label, text)
    return [
        {
            "label": label,
            "text": texts[label],
            "readings": {name: list(reader(texts[label])) for name, reader in READERS},
        }
        for label in sorted(texts)
    ]


def _agreeing_stored(limit: int = 2) -> list[dict]:
    """The first `limit` rows per log that Part A's filter accepts.

    Rng-free and head-only: chosen so the sample is reproducible from the file
    order alone and so regeneration does not re-read 8.5 GiB a second time. The
    filter is Part A's, copied from `_scan`: scored, on an LLM tier, non-empty
    `raw_text`. Line numbers count every physical line, blank ones included, so
    the labels are the script's own.
    """
    rows: list[dict] = []
    for path in _attempt_files():
        taken = 0
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if taken >= limit:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get("row_status")) != "scored":
                    continue
                if str(row.get("tier")) not in LLM_TIERS:
                    continue
                text = row.get("raw_text") or ""
                if not text:
                    continue
                rows.append({
                    "label": f"{path.parent.name}:{number}",
                    "text": text,
                    "reading": list(_LIVE_READER(text)),
                })
                taken += 1
    return rows


def _regenerate() -> None:
    """Build the whole fixture in memory, then write it once.

    Raises rather than writing a partial file when the corpus is absent: a
    fixture missing the population section would turn group (d) into a silent
    pass, which is the defect this module exists to close.
    """
    if not _CORPUS:
        raise AssertionError(
            "PARSER_DIFF_REGEN=1 requires data/comparison*/runs/*/attempts.jsonl"
        )
    result = run(examples_cap=10**9)
    data = {
        "_readme": _README,
        "_regenerate": _REGENERATE,
        "population": {
            "files": len(result["files"]),
            "stored": result["stored"].totals(),
            "mutants": result["mutants"].totals(),
            "census": dict(result["census"]),
        },
        # Read off the script's globals, never retyped: these three values ARE
        # the mutant population (15 logs x RESERVOIR x CUTS, seeded).
        "constants": {"seed": SEED, "reservoir": RESERVOIR, "cuts": CUTS},
        "divergent_mutants": _divergent_mutants(result),
        "agreeing_stored": _agreeing_stored(),
    }
    _GOLDENS.parent.mkdir(parents=True, exist_ok=True)
    _GOLDENS.write_text(
        json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def expected() -> dict:
    if os.environ.get("PARSER_DIFF_REGEN") == "1":
        _regenerate()
    if not _GOLDENS.exists():
        raise AssertionError(f"golden fixture missing: {_GOLDENS} (see _REGENERATE)")
    return json.loads(_GOLDENS.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Fixture shape
# --------------------------------------------------------------------------

def test_fixture_is_sorted_json_with_provenance(expected):
    raw = _GOLDENS.read_text(encoding="utf-8")
    assert raw == json.dumps(expected, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    assert expected["_readme"] == _README
    assert "2026-08-03" in expected["_readme"]
    assert "ccf53bf" in expected["_readme"]
    assert (
        "A diff here is a corpus or parser change, not a test to update."
        in expected["_readme"]
    )
    assert expected["_regenerate"] == _REGENERATE
    assert expected["_regenerate"].startswith("PARSER_DIFF_REGEN=1 ")


# --------------------------------------------------------------------------
# (a) The six divergent mutants — re-read here, no corpus needed
# --------------------------------------------------------------------------

def test_every_divergent_mutant_rereads_as_recorded(expected):
    """The fixture keeps each mutant's FULL prefix, so all four parsers run on
    the real input rather than on a report tail. The frozen ``new`` reading
    includes the grid projection present when the measurement was captured;
    its categorical coordinates remain the parser contract, while current
    scoring deliberately leaves the retired third coordinate absent."""
    entries = expected["divergent_mutants"]
    assert len(entries) == _MEASURED["old_live_ne_old_batch"] == 6
    assert len({entry["label"] for entry in entries}) == 6
    for entry in entries:
        for name, reader in READERS:
            actual = list(reader(entry["text"]))
            recorded = entry["readings"][name]
            if name == "new":
                assert actual[:2] == recorded[:2], (entry["label"], name)
                assert actual[2] is None, (entry["label"], name)
                assert recorded[2] == entry["readings"]["old_live"][2], (
                    entry["label"], name
                )
            else:
                assert actual == recorded, (entry["label"], name)


def test_the_divergence_is_the_batch_parser_losing_the_verdict(expected):
    """One shape, six times: the batch parser read ABSENCE where the live
    parser, the live baseline profile and the unified parser all read the same
    committed verdict. This is the divergence `indra_belief.verdict` resolves in
    the live parser's favour, and the reason Part B exists at all — the stored
    corpus cannot hold it, because `replay.py::error_row` writes `raw_text: ""`
    on every error row."""
    for entry in expected["divergent_mutants"]:
        readings = entry["readings"]
        assert readings["old_batch"] == [None, None, None], entry["label"]
        assert readings["new"] == readings["old_live"], entry["label"]
        assert readings["new"] == readings["old_live_baseline"], entry["label"]
        assert readings["new"][0] in ("correct", "incorrect"), entry["label"]
        assert readings["new"][1] == "medium", entry["label"]


def test_the_discriminator_is_a_modal_verdict_phrase(expected):
    """WHY the batch parser lost them, measured rather than asserted in prose.

    Its verdict patterns accept only `(?:is|=)` after the verdict word; the live
    ones also accept `should be` / `would be`. Every one of the six carries a
    modal phrase — all SIX, not five — and `tests/test_verdict_parser.py` covers
    no such case, which is why this file is where the phrase is pinned.
    """
    entries = expected["divergent_mutants"]
    matched = [entry["label"] for entry in entries
               if _MODAL_VERDICT_PHRASE.search(entry["text"])]
    assert matched == [entry["label"] for entry in entries]
    assert len(matched) == 6
    sibling = (_ROOT / "tests" / "test_verdict_parser.py").read_text(encoding="utf-8")
    assert "should be" not in sibling, (
        "tests/test_verdict_parser.py now covers a 'should be' verdict phrase; "
        "the discriminator this module pins may belong there instead"
    )


# --------------------------------------------------------------------------
# (b) Stored responses all four parsers agree on — no corpus needed
# --------------------------------------------------------------------------

def test_all_four_parsers_agree_on_the_stored_sample(expected):
    """Bit-rot detection on 29 texts, NOT population evidence.

    The population claim — exact agreement across 228,812 stored responses — is
    group (d)'s, and only the corpus can make it. These 29 are the first two
    accepted rows of each of the 15 logs (e2b_smoke has only one), chosen for
    determinism rather than coverage, and their diversity is thin BY
    CONSTRUCTION: 8 of the 29 are 44-character verdict-only replies of
    near-identical shape, because four of the logs come from the verdict-only
    arm where that is what a reply looks like. What this catches is a parser
    edit that changes an ordinary reading; what it cannot catch is a rare shape
    none of these 29 happens to have.
    """
    rows = expected["agreeing_stored"]
    assert len(rows) == 29
    assert len({row["label"] for row in rows}) == 29
    per_log = Counter(row["label"].split(":")[0] for row in rows)
    assert len(per_log) == _MEASURED["files"] == 15
    assert per_log["e2b_smoke"] == 1
    assert sorted(set(per_log.values())) == [1, 2]
    # The thinness, as a measured property rather than a claim in prose.
    assert sum(1 for row in rows if len(row["text"]) == 44) == 8
    for row in rows:
        readings = {name: reader(row["text"]) for name, reader in READERS}
        pairs = {name: tuple(value[:2]) for name, value in readings.items()}
        assert len(set(pairs.values())) == 1, (row["label"], pairs)
        assert list(readings["old_live"]) == row["reading"], row["label"]
        assert row["reading"][0] in ("correct", "incorrect"), row["label"]


# --------------------------------------------------------------------------
# (c) Constants, totals shape, and the prose that cites the measurement
# --------------------------------------------------------------------------

def test_the_seeded_constants_that_define_the_population(expected):
    """SEED / RESERVOIR / CUTS are the mutant population's definition: change one
    and Part B's 16,756 mutants and 11,645 off-grid scores are different numbers
    about a different sample."""
    assert expected["constants"] == {"seed": SEED, "reservoir": RESERVOIR, "cuts": CUTS}
    assert expected["constants"] == {"seed": 1234, "reservoir": 400, "cuts": 4}


def test_totals_keeps_every_pair_key_including_the_zeroes(expected):
    """The zeroes ARE the claim, so `totals()` builds its keys from PAIRS rather
    than handing back `dict(self.pair_counts)`, which would drop them."""
    wanted = {f"{left}!={right}" for left, right in PAIRS}
    assert set(Diff().totals()["pairs"]) == wanted
    for half in ("stored", "mutants"):
        assert set(expected["population"][half]["pairs"]) == wanted, half
    assert expected["population"]["stored"]["pairs"] == dict.fromkeys(wanted, 0)


def test_the_frozen_population_is_the_measured_one(expected):
    """The fixture against the numbers measured before it existed. Repeating them
    outside the file is what makes a silent re-baseline impossible."""
    population = expected["population"]
    stored, mutants = population["stored"], population["mutants"]
    assert population["files"] == _MEASURED["files"]
    assert stored["rows"] == _MEASURED["stored_rows"]
    assert stored["considered"] == _MEASURED["stored_considered"]
    assert stored["off_grid"] == 0
    assert set(stored["pairs"].values()) == {0}
    assert mutants["rows"] == _MEASURED["mutant_rows"]
    assert mutants["considered"] == _MEASURED["mutant_considered"]
    assert mutants["pairs"]["old_live!=old_batch"] == _MEASURED["old_live_ne_old_batch"]
    assert mutants["pairs"]["old_batch!=new"] == _MEASURED["old_batch_ne_new"]
    assert mutants["pairs"]["old_live!=new"] == 0
    assert mutants["pairs"]["old_live_baseline!=new"] == 0
    assert mutants["off_grid"] == _MEASURED["off_grid"]
    # rows == RESERVOIR x CUTS per log, minus the logs too small to fill it.
    assert mutants["considered"] == mutants["rows"]
    assert mutants["rows"] <= population["files"] * RESERVOIR * CUTS


def test_the_population_numbers_are_cited_in_prose(expected):
    """The three files that state these numbers, checked by literal substring.

    Not a duplicate of scripts/check_doc_anchors.py, which guards paths and
    dotted symbols and never a numeric claim. A failure here means a citation
    drifted from the frozen measurement — a human must reconcile the two. Do not
    weaken the assertion, and do not edit the fixture to match new prose.
    """
    considered = f"{expected['population']['stored']['considered']:,}"
    off_grid = f"{expected['population']['mutants']['off_grid']:,}"
    assert (considered, off_grid) == ("228,812", "11,645")
    for needle, names in ((considered, _CITES_CONSIDERED), (off_grid, _CITES_OFF_GRID)):
        for name in names:
            text = (_ROOT / name).read_text(encoding="utf-8")
            assert needle in text, (
                f"{name} no longer cites {needle}, the frozen measurement in "
                f"{_GOLDENS.relative_to(_ROOT)}. A citation has drifted from the "
                "evidence; reconcile them by hand."
            )


# --------------------------------------------------------------------------
# (d) The population itself — the corpus is the only thing that can prove it
# --------------------------------------------------------------------------

@requires_corpus
def test_the_whole_corpus_still_reads_the_same_way(expected):
    """Re-run the script's scan and require EXACT equality with the fixture.

    No tolerance, no approximation, no opt-out. This is the test that makes the
    228,812-response equivalence claim a thing the repository checks rather than
    a thing it remembers; a per-log `(size, mtime_ns)` comparison either side
    proves the scan left the published artifact untouched.
    """
    result = run()
    population = expected["population"]
    assert len(result["files"]) == population["files"]
    assert result["stored"].totals() == population["stored"]
    assert result["mutants"].totals() == population["mutants"]
    assert dict(result["census"]) == population["census"]
    assert _log_stats() == _STATS_AT_IMPORT, (
        "an attempts.jsonl changed size or mtime while this module ran"
    )
