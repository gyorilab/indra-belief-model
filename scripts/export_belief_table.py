#!/usr/bin/env python3
"""Emit the belief table a scored run produces: ``{matches_hash: belief}``.

WHY THIS SHAPE
--------------
Belief has exactly one origin in the INDRA stack and it is a ``{matches_hash:
float}`` table. CoGEx, INDRA DB, network search and MSstatsBioNet all read a
frozen copy of it — ``sort_by=belief`` and ``minimum_belief`` are queries against
that column. So deploying is not an integration project; it is a question of
which copy we write. This script writes ours.

Realtime scoring is not the alternative: a median statement takes 52 s to score
serially, and one real MSstatsBioNet subnetwork call returned 4,447 statements.
The table is the product; a lookup against it is the service.

THREE RULES, EACH PAID FOR
--------------------------
1. ABSENCE IS NOT 1.0. ``Statement.from_json`` silently defaults a MISSING
   belief to 1.0, so emitting a placeholder for a statement we could not read
   would publish "certainly true". Unscored statements are OMITTED and counted,
   never defaulted.

2. PARTIAL COVERAGE BREAKS RANKING. A list mixing our belief with the
   incumbent's is ranking on two different scales — a correctness bug, not a
   coverage gap. The manifest states coverage explicitly so a consumer can
   refuse a partial table rather than silently blend.

3. THE PROFILE TRAVELS WITH THE NUMBER. A belief is meaningless without the
   calibration that produced it: the same reader on a different prompt is a
   different instrument. ``profile_id`` and the prompt digest are recorded, and
   an uncalibrated run is labelled as such rather than passed off as fitted.

Usage:
    python scripts/export_belief_table.py \
        --run data/results/<run>.jsonl \
        --corpus data/corpora/<corpus>.json \
        --out data/belief_tables/<name>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.results import build_run_export  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_table(per_stmt: list[dict]) -> tuple[dict[str, float], dict]:
    """Project per-statement records onto the INDRA belief column.

    Returns the table and a diagnostics block. Collisions are reported rather
    than resolved: two statements sharing a matches_hash with DIFFERENT beliefs
    is a join defect upstream, and last-writer-wins would bury it.
    """
    table: dict[str, float] = {}
    seen: dict[str, set] = {}
    unscored = 0
    no_hash = 0
    for row in per_stmt:
        mh = row.get("indra_matches_hash")
        belief = row.get("belief")
        if mh is None:
            no_hash += 1
            continue
        if belief is None:
            # Rule 1: nothing was read. Omit; never default.
            unscored += 1
            continue
        key = str(mh)
        seen.setdefault(key, set()).add(round(float(belief), 9))
        table[key] = float(belief)
    collisions = {k: sorted(v) for k, v in seen.items() if len(v) > 1}
    return table, {
        "n_statements_in_run": len(per_stmt),
        "n_published": len(table),
        "n_unscored_omitted": unscored,
        "n_without_matches_hash": no_hash,
        "n_hash_collisions_with_differing_belief": len(collisions),
        "collision_examples": dict(list(collisions.items())[:5]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--gold", default=None, help="optional; only affects diagnostics")
    ap.add_argument("--model", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-calibrated", action="store_true",
                    help="exit non-zero if the run resolves no ship-approved profile")
    args = ap.parse_args()

    run_path = Path(args.run)
    per_ev, per_stmt, meta, metrics = build_run_export(
        str(run_path), corpus_path=args.corpus, run_id=args.run_id,
        model=args.model, gold_path=args.gold,
    )
    table, diagnostics = build_table(per_stmt)

    # The resolved profile lives at metrics_basis.soft_calibration — NOT at a
    # top-level "soft_weights" key, which is always None. Reading the wrong key
    # labelled a calibrated table "HARD GATE", which is exactly the provenance
    # error this manifest exists to prevent, so the lookup is asserted below.
    soft = ((metrics or {}).get("metrics_basis") or {}).get("soft_calibration") or {}
    profile_id = soft.get("profile_id")
    calibrated = soft.get("status") == "available" and bool(profile_id)
    reader = {
        "model": (soft.get("reader_configuration") or "").split("@prompt-sha256:")[0] or None,
        "prompt_sha256": ((soft.get("reader_configuration") or "").split("@prompt-sha256:") + [None])[1],
    }
    # A run whose beliefs moved off the hard gate MUST name the profile that moved
    # them. Publishing a fitted number under a "HARD GATE" label is worse than
    # publishing no label.
    moved = any(r.get("belief_soft") is not None and r.get("belief_soft") != r.get("belief_hard")
                for r in per_stmt)
    if moved and not calibrated:
        raise SystemExit(
            "refusing to write: beliefs differ from the hard gate but no profile "
            "resolved — the table would be mislabelled as uncalibrated"
        )

    incumbent = sum(1 for r in per_stmt if r.get("rasmachine_belief") is not None)
    manifest = {
        "kind": "indra_belief_table",
        "schema_version": 1,
        "belief_column": "matches_hash -> belief (this project's score)",
        "calibrated": calibrated,
        "profile_id": profile_id,
        "reader_model": reader.get("model"),
        "prompt_sha256": reader.get("prompt_sha256"),
        "source_run": str(run_path),
        "source_run_sha256": _sha256(run_path),
        "corpus": args.corpus,
        "coverage": diagnostics,
        "n_with_incumbent_belief": incumbent,
        "consumer_notes": [
            "Unscored statements are ABSENT from the table. Do not default a "
            "missing key to 1.0 — INDRA's Statement.from_json does, and that "
            "publishes 'certainly true' for something never read.",
            "Do not mix this column with the incumbent belief in one ranked "
            "list: the two are different scales and the ordering would be "
            "meaningless. Rank within one column or not at all.",
            ("This run resolved no ship-approved calibration; beliefs are the "
             "HARD GATE fallback, not the fitted reader." if not calibrated else
             "Beliefs use the ship-approved profile named in profile_id; a "
             "different model or prompt is a different instrument."),
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    man = out.with_suffix(".manifest.json")
    man.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"  wrote {out}  ({len(table)} statements)")
    print(f"  wrote {man}")
    print(f"  calibrated: {calibrated}  profile: {profile_id or 'HARD GATE'}")
    for k, v in diagnostics.items():
        if k != "collision_examples":
            print(f"    {k}: {v}")
    if args.require_calibrated and not calibrated:
        print("  REFUSING: --require-calibrated set but this run has no fitted profile",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
