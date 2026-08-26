"""Assemble the multi-curator gold: your anchor + a curator-spread sample over the
other 37, balanced 1:1 and de-skewed by stmt_type.

Pipeline:
  1. gold rows from the re-fetched evidence (build_rasmachine_eval.gold_rows_for over
     recovered_curation_statements.json + every non-ben/bachman curation).
  2. DE-CONTAMINATE: drop any (mh,sh) in a prior holdout/eval/fewshot/benchmark/
     external_gold set.
  3. EQUAL CAP: each curator contributes at most C per class (default 40), attributed
     to its primary curator — so marta's SIGNOR pile can't capture the gold. This is
     the "random sample over other curators" that diversifies curation DIRECTION.
  4. BALANCED + DE-SKEWED: stratified 1:1 by stmt_type (build_curation_eval's rule —
     identical type marginal across classes). mock7ee is pooled as one curator.

Outputs external_curator_gold_v1.jsonl + its trimmed statements JSON (one scoring
call per pair). Contamination-guarded at the end.

    PYTHONPATH=src .venv/bin/python scripts/build_multicurator_gold.py [--cap 40]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from indra.statements import stmts_from_json, stmts_to_json  # noqa: E402

from build_curation_eval import fewshot_filters, is_contaminated, stratified_balanced  # noqa: E402
from build_rasmachine_eval import gold_rows_for  # noqa: E402
from indra_belief.curation import Curation, build_index  # noqa: E402

BENCH = ROOT / "data" / "benchmark"
CORPUS = ROOT / "data" / "corpora" / "recovered_curation_statements.json"
UNIVERSE = ROOT / "data" / "results" / "curation_universe_all.jsonl"
DEFAULT_NAME = "external_curator_gold_v1"


def output_paths(name: str) -> tuple[Path, Path]:
    """Gold rows and their trimmed statements, for one gold NAME.

    Parameterised because this script used to write two hardcoded paths, so the
    only way to build a bigger or differently-capped gold was to overwrite a
    tracked artifact that other runs are pinned against.

    Building under a NEW name is also automatically disjoint from the old one:
    `excl_pairs` scans every prior benchmark file and skips only the output
    being written, so a v2 excludes v1's pairs and the two can serve as
    independent fit and validation sets over the same population.
    """
    return (BENCH / f"{name}.jsonl",
            ROOT / "data" / "corpora" / f"{name}_statements.json")


GOLD_OUT, STMTS_OUT = output_paths(DEFAULT_NAME)
PAIRS = {"ben.gyori@gmail.com", "bachmanjohn@gmail.com", "ben.gyori", "bachmanjohn"}
SEED = 20260630


def excl_pairs(gold_out: Path | None = None) -> set[tuple[int, int]]:
    out = set()
    # EVERY benchmark file, not a hand-maintained pattern list. The list named
    # nine patterns and left SEVEN files uncovered -- including
    # representative_indra_curations_400.jsonl, the curated snapshot reserved
    # from the CoGEx representative pool. Two of its pairs duly appeared in a
    # freshly built gold, caught only by an unrelated test that audits that pool
    # against the whole benchmark directory.
    #
    # This repo already learned the lesson elsewhere: a hand-maintained list
    # makes "unguarded" the DEFAULT for a new file, and a glob inverts it. A
    # new benchmark set is now excluded-from on the commit that adds it.
    for p in sorted(BENCH.glob("*.jsonl")):
            if p in (gold_out or GOLD_OUT,):
                continue
            for line in p.open():
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                # BOTH identities, not whichever is present first. This read
                # `r.get("pa_hash", r.get("matches_hash"))`, preferring pa_hash
                # -- but the pipeline joins on matches_hash, and MEASURED on
                # eval_curation_v1 the two differ for 105 of 1606 rows. Those
                # pairs were excluded under an identity nothing downstream uses,
                # so they stayed eligible for a new gold: de-contamination that
                # keys on a different identity than the join is not
                # de-contamination. A pair is excluded if EITHER hash matches.
                source_hash = r.get("source_hash")
                if source_hash is None:
                    continue
                for stmt_hash in (r.get("pa_hash"), r.get("matches_hash")):
                    if stmt_hash is None:
                        continue
                    try:
                        out.add((int(stmt_hash), int(source_hash)))
                    except (TypeError, ValueError):
                        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=40, help="max pairs per curator PER CLASS")
    ap.add_argument("--name", default=DEFAULT_NAME,
                    help="gold NAME; writes data/benchmark/<name>.jsonl and "
                         "data/corpora/<name>_statements.json. A new name is "
                         "de-contaminated against the existing golds, so it can "
                         "serve as an independent fit set beside them")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the size a cap would yield and write nothing")
    args = ap.parse_args()
    gold_out, stmts_out = output_paths(args.name)
    rng = random.Random(SEED)

    # 1. gold rows from re-fetched evidence
    curs = []
    for line in UNIVERSE.open():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if (d.get("curator") or "").strip() in PAIRS or not d.get("curator"):
            continue
        c = Curation.from_dict(d)
        if c is not None:
            curs.append(c)
    idx = build_index(curs)
    stmts = stmts_from_json(json.load(open(CORPUS)))
    rows = gold_rows_for(stmts, idx)
    print(f"recovered gold rows: {len(rows)}")

    # 2. de-contaminate — by (mh,sh) pair AND by the inference-time fewshot universe
    # (text-norm + entity-pair), mirroring build_curation_eval so the guard passes.
    excl = excl_pairs(gold_out)
    rows = [r for r in rows if (r["matches_hash"], r["source_hash"]) not in excl]
    ex_pairs, ex_norms = fewshot_filters()
    before = len(rows)
    rows = [r for r in rows if not is_contaminated(r, ex_pairs, ex_norms)]
    print(f"after de-contamination (excl {len(excl)} prior-set pairs + "
          f"{before - len(rows)} fewshot-universe text/entity leaks): {len(rows)}")

    # 3. equal per-curator cap (attributed to primary curator), per class
    rng.shuffle(rows)
    seen = defaultdict(int)
    capped = []
    for r in rows:
        cur = (r.get("curators") or ["?"])[0]
        key = (cur, r["gold"])
        if seen[key] < args.cap:
            seen[key] += 1
            capped.append(r)
    n_cur = len({c for (c, _), v in seen.items() if v})
    print(f"after equal cap (<= {args.cap}/class/curator): {len(capped)} rows across {n_cur} curators")

    # 4. balanced 1:1 + de-skewed by stmt_type
    sample, report = stratified_balanced(capped, rng)
    rng.shuffle(sample)
    ni = sum(1 for r in sample if r["gold"] == "incorrect")
    nc = len(sample) - ni
    print(f"\nFINAL gold: {len(sample)} pairs ({nc} correct / {ni} incorrect)")
    print(f"  types used: {report['types_used']}  dropped(single-class): {report['types_dropped']}")
    print(f"  per-type each-class: {report['per_type_pairs_each_class']}")
    spread = Counter((r.get('curators') or ['?'])[0] for r in sample)
    print(f"  curator spread: {len(spread)} curators; {dict(spread.most_common(15))}")
    mh_anchor = sum(1 for r in sample if 'mock7ee@gmail.com' in (r.get('curators') or []))
    print(f"  your (mock7ee) anchor pairs in gold: {mh_anchor}")

    if args.dry_run:
        print(f"\n[dry-run] cap={args.cap} would yield {len(sample)} gold rows "
              f"-> {gold_out.name}; nothing written")
        return 0

    # write gold + trimmed statements
    gold_pairs = {(r["matches_hash"], r["source_hash"]) for r in sample}
    gold_mh = {mh for mh, _ in gold_pairs}
    with open(gold_out, "w") as f:
        for r in sample:
            f.write(json.dumps(r, default=str) + "\n")
    kept = []
    for s in stmts:
        mh = s.get_hash(refresh=True)
        if mh not in gold_mh:
            continue
        s.evidence = [ev for ev in (s.evidence or []) if (mh, ev.get_source_hash()) in gold_pairs]
        kept.append(s)
    json.dump(stmts_to_json(kept), open(stmts_out, "w"))
    print(f"\nwrote {len(sample)} gold rows -> {gold_out}")
    print(f"wrote {len(kept)} statements ({sum(len(s.evidence or []) for s in kept)} ev to score) -> {stmts_out}")

    # contamination guard
    try:
        import check_contamination as cc
        contam = cc.find_contamination(eval_paths=[gold_out])
        print(f"contamination guard: {'CLEAN ✓' if not contam else f'{len(contam)} OVERLAP(S)!'}")
    except Exception as e:  # noqa: BLE001
        print(f"contamination guard skipped: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
