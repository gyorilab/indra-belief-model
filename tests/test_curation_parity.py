"""The gold rule of src/indra_belief/curation.py, over the corner cases that
matter: the gold rule itself, any-incorrect-wins aggregation, the hash-bridge,
and the index shape.

This file used to be a cross-language parity guard. The curation domain was
deliberately duplicated in two languages — curation.py here and curation.ts in
the viewer — and a parity test drove both over one fixture so neither could
drift. The viewer was removed, taking curation.ts and its
its Node runner with it, so the duplication the
guard existed to protect is gone and the parity test was deleted with it.
Python is now the single implementation; the fixture below still pins its
behaviour.
"""
from __future__ import annotations

from pathlib import Path

from indra_belief.curation import build_index, Curation

ROOT = Path(__file__).resolve().parents[1]

# Fixture exercising the corners the gold rule must get right:
#  - single 'correct'                      -> correct
#  - single dissenting tag                 -> incorrect
#  - multi-curation all correct            -> correct
#  - multi-curation one dissent (any-wins) -> incorrect
#  - duplicate curator (deduped in curators list)
#  - empty + whitespace notes (filtered out)
#  - negative int hash + string-ish int
FIXTURE = [
    {"_matches_hash": 100, "source_hash": 1, "tag": "correct", "curator": "a@x", "date": "d1", "text": ""},
    {"_matches_hash": 100, "source_hash": 2, "tag": "grounding", "curator": "b@x", "date": "d2", "text": "wrong gene"},
    {"_matches_hash": 200, "source_hash": 3, "tag": "correct", "curator": "a@x", "date": "d3", "text": "  "},
    {"_matches_hash": 200, "source_hash": 3, "tag": "correct", "curator": "a@x", "date": "d4", "text": "ok"},
    {"_matches_hash": 300, "source_hash": 4, "tag": "correct", "curator": "c@x", "date": "d5", "text": ""},
    {"_matches_hash": 300, "source_hash": 4, "tag": "wrong_relation", "curator": "d@x", "date": "d6", "text": "flip"},
    {"_matches_hash": -42, "source_hash": -7, "tag": "polarity", "curator": "e@x", "date": "d7", "text": "neg"},
]


def _py_gold(fixture: list[dict]) -> dict:
    curs = [c for c in (Curation.from_dict(d) for d in fixture) if c is not None]
    index = build_index(curs)
    gold = {}
    for (mh, sh), gv in sorted(index.gold_by_key.items()):
        gold[f"{mh}|{sh}"] = {
            "verdict": gv.verdict,
            "n": gv.n,
            "tags": gv.tags,
            "curators": gv.curators,
            "notes": gv.notes,
        }
    return {"n_statements": index.n_statements, "n_evidences": index.n_evidences, "gold": gold}


def test_python_gold_rule():
    """Lock the gold rule independently of the parity hop."""
    g = _py_gold(FIXTURE)["gold"]
    assert g["100|1"]["verdict"] == "correct"
    assert g["100|2"]["verdict"] == "incorrect"  # grounding
    assert g["200|3"]["verdict"] == "correct" and g["200|3"]["n"] == 2  # both correct
    assert g["200|3"]["curators"] == ["a@x"]  # deduped
    assert g["200|3"]["notes"] == ["ok"]  # whitespace-only filtered
    assert g["300|4"]["verdict"] == "incorrect"  # any-incorrect-wins
    assert g["-42|-7"]["verdict"] == "incorrect"  # polarity, negative hashes


