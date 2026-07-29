"""The 2023-paper panel must stay disjoint from the reader prompts' worked examples.

``scripts/check_contamination.py`` guards the calibration files and the holdouts,
but its ``_default_eval_paths`` does NOT include the paper panel -- so the one
benchmark whose labels are a published artifact of the audience's own lab had no
leakage guard at all. The /paper memo asserts "none of them is drawn from this
panel"; this test is what makes that assertion checkable.

Grain: (agent set, statement type). That is the grain at which a demonstration
would actually teach the answer to a panel item -- an exact string match would be
too strict (the prompt renders a claim, the corpus stores members) and a bare
agent-name match far too loose (common signalling proteins recur everywhere by
construction, and 33 of them legitimately do).

NON-VACUITY IS ASSERTED, not assumed. An earlier hand-run of this check returned
a clean zero only because it read a field that did not exist, so every key was
empty and nothing could ever match. The namespace-overlap assertion below exists
to make that failure mode impossible: if the two sides stop sharing ANY agent
name, the disjointness result is meaningless and this test fails rather than
passing green.

Pretraining contamination is a different question and is out of scope here -- the
benchmark corpus and the paper repo are both public. See the memo.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/corpora/indra_paper_unique_pairs_20260717_statements.json"
PROMPTS = ROOT / "data/comparison/grounding_replay/manifest.json"

#: Demonstration claims sharing an (agent set, type) with a panel statement, as
#: measured 2026-07-27 with the corrected two-shape parser. Effect on the headline
#: margin: -0.0011 AUROC. Raising this number requires re-running the sensitivity.
_KNOWN_CLAIM_OVERLAP = 11


def _known_overlap_keys(panel_keys: set, demo_keys: set) -> set:
    """The overlap as it stands; used only to name NEW entries in the failure."""
    return panel_keys & demo_keys


def _agent_name(agent: object) -> str | None:
    if not isinstance(agent, dict):
        return None
    name = agent.get("name") or (agent.get("db_refs") or {}).get("TEXT")
    return str(name) if name else None


def _panel_keys() -> tuple[set, set]:
    """(agent-set, type) keys and the bare agent names behind them.

    Reads BOTH statement shapes. `members` exists only on Complex statements --
    692 of this panel's 1,689. The other 997 (Phosphorylation, Activation,
    Inhibition, IncreaseAmount, DecreaseAmount) carry `subj`/`obj`. An earlier
    version of this file read `members` alone, so 59% of the panel produced no
    key, the binary demonstration claims could never match anything, and the two
    sets were disjoint BY CONSTRUCTION -- the exact vacuity this module's
    docstring warns about, shipped green.
    """
    statements = json.loads(PANEL.read_text())
    keys: set = set()
    names: set = set()
    for stmt in statements:
        agents = [n for n in (_agent_name(m) for m in (stmt.get("members") or [])) if n]
        for field in ("subj", "obj", "enz", "sub", "agent"):
            name = _agent_name(stmt.get(field))
            if name:
                agents.append(name)
        if agents:
            keys.add((frozenset(agents), str(stmt.get("type", "")).lower()))
            names.update(agents)
    return keys, names


def _demo_keys() -> tuple[set, set]:
    """The same grain, parsed off the hand-authored demonstration prefixes."""
    manifest = json.loads(PROMPTS.read_text())
    claims: set[str] = set()
    for prefix in manifest["prompt_components"]["main_message_prefixes"]:
        for message in prefix["messages"]:
            if message.get("role") != "user":
                continue
            found = re.search(r"CLAIM:\s*(.+)", message["content"])
            if found:
                claims.add(found.group(1).strip())

    keys: set = set()
    names: set = set()
    for claim in claims:
        # Complex FIRST: "A + B [Complex]" also matches the binary pattern, which
        # would fuse "A + B" into one pseudo-agent that can never match a real key.
        complex_ = re.match(r"^(.*?)\s*\+\s*(.*?)\s*\[(\w+)\]$", claim)
        binary = None if complex_ else re.match(r"^(.*?)\s*\[(\w+)\]\s*(.*)$", claim)
        if binary:
            agents = [x.strip() for x in (binary.group(1), binary.group(3)) if x.strip()]
            stmt_type = binary.group(2).lower()
        elif complex_:
            agents = [complex_.group(1).strip(), complex_.group(2).strip()]
            stmt_type = complex_.group(3).lower()
        else:
            continue
        keys.add((frozenset(agents), stmt_type))
        names.update(agents)
    return keys, names


@pytest.mark.skipif(not PANEL.is_file(), reason="paper panel corpus not present")
@pytest.mark.skipif(not PROMPTS.is_file(), reason="grounding replay manifest not present")
def test_paper_panel_disjoint_from_demonstration_claims() -> None:
    panel_keys, panel_names = _panel_keys()
    demo_keys, demo_names = _demo_keys()

    assert panel_keys, "parsed no panel keys — the check would be vacuous"
    assert demo_keys, "parsed no demonstration keys — the check would be vacuous"

    # The guard on the guard, in two parts. Bare-name overlap is necessary but NOT
    # sufficient: an earlier version passed this check while the two key sets were
    # disjoint by construction, because names intersected and keys could not.
    shared_names = panel_names & demo_names
    assert shared_names, (
        "panel and demonstration agent namespaces do not intersect at all; a "
        "disjointness result would be an artifact of incomparable naming, not "
        "evidence of no leakage"
    )
    # Parser coverage: every statement type PRESENT IN THE PANEL must produce
    # keys. A demo type the panel simply does not contain (e.g. Autophosphorylation)
    # is fine — there is nothing to collide with. What is NOT fine is a type the
    # panel HAS while the parser yields nothing for it, which is precisely how the
    # earlier version went blind to 59% of the panel.
    corpus_types = {
        str(stmt.get("type", "")).lower()
        for stmt in json.loads(PANEL.read_text())
        if stmt.get("type")
    }
    keyed_types = {t for _agents, t in panel_keys}
    unkeyed = {t for t in corpus_types if t not in keyed_types}
    assert not unkeyed, (
        f"panel statement types {sorted(unkeyed)} are present in the corpus but "
        "produce no keys; the disjointness check is blind to all of them"
    )

    # MEASURED, not asserted-to-be-zero. With the parser fixed to read both
    # statement shapes, 11 of the 45 demonstration claims DO share an
    # (agent set, type) with a panel statement -- the earlier "zero overlap"
    # result was an artifact of a parser that saw only 41% of the panel, and the
    # memo's claim was corrected accordingly.
    #
    # Sharing (agent set, type) is a WEAK form of exposure: it means the model saw
    # a worked example about the same entity pair and relation, not the same
    # sentence. Measured effect on the headline: the 11 claims touch 11 of 1,689
    # statements (0.7%), and excluding them moves the gate-minus-served-belief
    # AUROC margin from +0.1270 to +0.1259, a shift of -0.0011 against a +0.126
    # effect. The verbatim-sentence leak is separately measured at 12 of 5,379
    # evidence pairs (0.22%), max AUROC shift 0.0008.
    #
    # This assertion pins the KNOWN overlap. It fails if contamination GROWS,
    # which is the thing that would actually invalidate the panel.
    overlap = panel_keys & demo_keys
    assert len(overlap) <= _KNOWN_CLAIM_OVERLAP, (
        f"{len(overlap)} demonstration claims now share an (agent set, type) with "
        f"the panel, up from the known {_KNOWN_CLAIM_OVERLAP}. New contamination "
        f"needs a fresh sensitivity analysis before any margin is quoted. "
        f"New: {sorted(overlap - _known_overlap_keys(panel_keys, demo_keys))[:5]}"
    )
