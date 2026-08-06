#!/usr/bin/env python
"""Measure the modularity of the two scoring paths — the BEFORE/AFTER instrument.

Answers "is this an intuitive codebase?" with numbers instead of opinion:

  A. path_modules      distinct indra_belief.* modules whose code EXECUTES while
                       one (statement, evidence) pair becomes one float, traced
                       with sys.settrace. Measured for the live path and the
                       batch path separately, plus the union and the shared set.
  B. shared_fraction   |live ∩ batch| / |live ∪ batch|. One semantic kernel means
                       this goes UP; two implementations keep it near zero.
  C. api_surface       module-level public names (no leading underscore) exported
                       by src/indra_belief/**.py, counted with ast — the reader's
                       vocabulary.
  D. duplicate_sites   the enumerated (concept, live_anchor, batch_anchor) table.
                       Each row is a semantic decision implemented twice.
  E. concepts          distinct named data shapes one score is re-encoded into
                       between input pair and published number.

Run:  PYTHONPATH=src .venv/bin/python scripts/modularity_baseline.py
Emits canonical JSON on stdout. Diff a BEFORE capture against an AFTER capture;
`--check BEFORE.json` exits non-zero on a regression (see __main__).
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "indra_belief"


# ── A. dynamic module traversal ──────────────────────────────────────────────

class _Trace:
    def __init__(self) -> None:
        self.modules: set[str] = set()

    def __enter__(self):
        sys.settrace(self._hook)
        return self

    def __exit__(self, *exc):
        sys.settrace(None)
        return False

    def _hook(self, frame, event, arg):
        filename = frame.f_code.co_filename
        if SRC.as_posix() in filename:
            rel = Path(filename).resolve().relative_to(SRC.parent)
            self.modules.add(rel.with_suffix("").as_posix().replace("/", "."))
        return self._hook


class _StubResponse:
    content = '{"relation_check": "direct", "support": "s", "objection": null, "verdict": "correct", "confidence": "high"}'
    raw_text = content
    tokens = 10
    reasoning_trace = None


class _StubClient:
    """Satisfies both ModelClient duck-typing and replay.ClientLike."""

    def call(self, *args, **kwargs):
        return _StubResponse()

    def pop_call_log(self):
        return []


def trace_live() -> set[str]:
    from indra.statements import Agent, Evidence, Phosphorylation

    from indra_belief.data.scoring_record import ScoringRecord
    from indra_belief.scorers.monolithic import scorer

    subject = Agent("MAP2K1", db_refs={"HGNC": "6840"})
    object_ = Agent("MAPK1", db_refs={"HGNC": "6871"})
    evidence = Evidence(
        source_api="reach",
        text="MAP2K1 phosphorylates MAPK1 in vitro.",
        annotations={"agents": {"raw_text": ["MEK1", "ERK2"],
                                "agent_list": ["MAP2K1", "MAPK1"]}},
    )
    record = ScoringRecord(statement=Phosphorylation(subject, object_), evidence=evidence)
    with _Trace() as t:
        scorer.score(_StubClient(), record, max_tokens=64)
    return t.modules


def trace_batch() -> set[str]:
    from indra_belief.comparison import replay

    system = "SYSTEM"
    prefix: tuple[dict[str, str], ...] = ()
    row = {
        "route": "plain",
        "claim": "MAP2K1 [Phosphorylation] MAPK1",
        "entity_context": "Entities: MAP2K1 | MAPK1",
        "abbreviation_lines": [],
        "provenance": "",
        "lookup_refs": [],
        "evidence_metadata": {"text": "MAP2K1 phosphorylates MAPK1 in vitro."},
        "main_system_ref": replay._sha_text(system),
        "main_message_prefix_ref": replay.canonical_sha256(list(prefix)),
        "subject_name": "MAP2K1",
        "object_name": "MAPK1",
    }
    index = replay.ReplayIndex(
        manifest={}, captures=(), systems={row["main_system_ref"]: system},
        prefixes={row["main_message_prefix_ref"]: prefix}, entities={},
        lookups={}, relation_aliases={}, executions=(row,),
    )
    row["main_prompt_base_sha256"] = index.prepare(row).calls()[-1].prompt_sha256()
    with _Trace() as t:
        replay.score_execution(index, row, _StubClient(), main_max_tokens=64)
    return t.modules


# ── C. public API surface ────────────────────────────────────────────────────

def api_surface() -> dict[str, int]:
    per_module: dict[str, int] = {}
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        names: set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not node.name.startswith("_"):
                    names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and not target.id.startswith("_"):
                        names.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if not node.target.id.startswith("_"):
                    names.add(node.target.id)
        rel = path.relative_to(SRC.parent).with_suffix("").as_posix().replace("/", ".")
        per_module[rel] = len(names)
    return per_module


# ── D. duplicated-logic sites ────────────────────────────────────────────────
# Each row: one semantic decision with TWO implementations. A SOLID change that
# removes a concept empties rows here; a change that adds a layer does not.
DUPLICATE_SITES = [
    # RETIRED by K1-prepared-execution: the user message is rendered once, by
    # `prepared_execution.ExecutionBody.render`, which both sides produce into.
    #
    # RETIRED by K2-one-parser, four rows — "structured verdict parse",
    # "phrase-level verdict fallback", "(verdict, confidence) -> score grid" and
    # "score on unparseable output". `indra_belief.verdict` is the one reader and
    # the one map; the score grid itself stays in `scorers/_shared.py` (its bytes
    # feed a published implementation_digest) and is imported, not copied. The
    # last of the four was not merely duplicated but CONTRADICTORY — live wrote
    # 0.5, batch wrote None — and it resolved to None: an absent measurement
    # stays absent.
    #
    # RETIRED by X1-twins, two rows — "relation-nature note" and "no-text
    # default-accept". The retired relation anchors named the two
    # `_relation_note` wrappers, which are not twins and remain: live dispatches
    # a call while batch formats a stored reply. What collapsed is the label
    # table, mismatch sentence, and relation user message, now owned by
    # `prepared_execution.relation_mismatch_note` /
    # `prepared_execution.relation_user_message`, plus the immutable
    # default-accept core now owned by `verdict.NO_TEXT_RESULT`.
    {"concept": "provenance / grounding gate",
     "live": "src/indra_belief/data/scoring_record.py:369 recomputed from has_grounding_signal",
     "batch": "src/indra_belief/comparison/replay.py:383 trusts row['provenance']"},
    {"concept": "few-shot prefix selection",
     "live": "src/indra_belief/scorers/monolithic/scorer.py _select_examples at call time",
     "batch": "src/indra_belief/comparison/replay.py:394 frozen prefix by sha256"},
    {"concept": "deterministic auto-reject",
     "live": "src/indra_belief/data/scoring_record.py:392 tier1_auto_reject",
     "batch": "src/indra_belief/comparison/replay.py:447 deterministic_result"},
]

# ── E. concepts a reader must hold on the score path ─────────────────────────
CONCEPTS = [
    "indra.Statement + indra.Evidence (the input pair)",
    "ScoringRecord (resolved pair, live)",
    "GroundedEntity (grounding verdict)",
    "execution row (the batch shard record)",
    "ReplayIndex (loader + validator + hydrator + renderer)",
    "prompt component refs (main_system_ref / main_message_prefix_ref)",
    "source row (runner input)",
    "result row / error row (runner output)",
    "ResumeState (crash-safe resume)",
    "attempt row (spend_guard durable state)",
    "ledger event (spend_guard reservation lifecycle)",
    "reservation (spend_guard in-memory)",
    "run plan / action / stage / workload (contracts)",
    "EvidenceObservation-equivalent (per-evidence verdict dict)",
    "statement belief (aggregated scalar)",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", type=Path, default=None,
                        help="BEFORE json; exit 1 if AFTER regresses")
    args = parser.parse_args()

    live, batch = trace_live(), trace_batch()
    surface = api_surface()
    report = {
        "path_modules": {
            "live": sorted(live),
            "batch": sorted(batch),
            "live_count": len(live),
            "batch_count": len(batch),
            "shared": sorted(live & batch),
            "shared_count": len(live & batch),
            "union_count": len(live | batch),
            "shared_fraction": round(len(live & batch) / max(len(live | batch), 1), 4),
        },
        "api_surface": {
            "total": sum(surface.values()),
            "modules": len(surface),
            "per_module": surface,
        },
        "duplicate_sites": DUPLICATE_SITES,
        "duplicate_site_count": len(DUPLICATE_SITES),
        "concepts": CONCEPTS,
        "concept_count": len(CONCEPTS),
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if args.check:
        before = json.loads(args.check.read_text())
        fails = []
        if report["duplicate_site_count"] > before["duplicate_site_count"]:
            fails.append("duplicate_site_count rose")
        if report["concept_count"] > before["concept_count"]:
            fails.append("concept_count rose")
        if report["api_surface"]["total"] > before["api_surface"]["total"]:
            fails.append("api_surface total rose")
        if report["path_modules"]["shared_fraction"] < before["path_modules"]["shared_fraction"]:
            fails.append("shared_fraction fell")
        if fails:
            print("CEREMONY: " + "; ".join(fails), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
