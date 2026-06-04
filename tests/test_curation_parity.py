"""Cross-language parity: src/indra_belief/curation.py vs the viewer's
viewer/src/lib/data/curation.ts must reduce the same curations JSONL to the same
gold index. This is the drift guard for the deliberate two-language duplication
of the curation domain — neither side may diverge on the gold rule, the
any-incorrect-wins aggregation, the hash-bridge, or the index shape.

The TS side runs via Node's native type-stripping (node --experimental-strip-types),
through viewer/scripts/curation_gold_json.mjs. Skipped if node is unavailable.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from indra_belief.curation import build_index, Curation

ROOT = Path(__file__).resolve().parents[1]
TS_RUNNER = ROOT / "viewer" / "scripts" / "curation_gold_json.mjs"

# Fixture exercising the corners the two languages must agree on:
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


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_curation_gold_parity(tmp_path):
    """Python and TypeScript must produce byte-identical gold from one fixture."""
    fixture_path = tmp_path / "curations.jsonl"
    fixture_path.write_text("\n".join(json.dumps(c) for c in FIXTURE) + "\n")

    py = _py_gold(FIXTURE)

    proc = subprocess.run(
        ["node", "--experimental-strip-types", str(TS_RUNNER), str(fixture_path)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"TS runner failed: {proc.stderr}"
    ts = json.loads(proc.stdout)

    assert ts == py, (
        "Python and TypeScript curation gold DIVERGED.\n"
        f"  python: {json.dumps(py, sort_keys=True)}\n"
        f"  ts:     {json.dumps(ts, sort_keys=True)}"
    )
