"""Assemble the multi-curator gold: your anchor + a curator-spread sample over the
other 37, balanced 1:1 and de-skewed by stmt_type.

Pipeline (knobs chosen with the user 2026-06-30):
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
GOLD_OUT = BENCH / "external_curator_gold_v1.jsonl"
STMTS_OUT = ROOT / "data" / "corpora" / "external_curator_gold_v1_statements.json"
PAIRS = {"ben.gyori@gmail.com", "bachmanjohn@gmail.com", "ben.gyori", "bachmanjohn"}
SEED = 20260630


def excl_pairs() -> set[tuple[int, int]]:
    out = set()
    pats = ("holdout*.jsonl", "eval_set*.jsonl", "eval_curation*.jsonl",
            "fewshot*.jsonl", "belief_benchmark.jsonl", "external_gold*.jsonl",
            "external_curator*.jsonl", "rasmachine*.jsonl", "probe_*.jsonl")
    for pat in pats:
        for p in BENCH.glob(pat):
            if p in (GOLD_OUT,):
                continue
            for line in p.open():
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                a, b = r.get("pa_hash", r.get("matches_hash")), r.get("source_hash")
                if a is not None and b is not None:
                    try:
                        out.add((int(a), int(b)))
                    except (TypeError, ValueError):
                        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=40, help="max pairs per curator PER CLASS")
    args = ap.parse_args()
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
    excl = excl_pairs()
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

    # write gold + trimmed statements
    gold_pairs = {(r["matches_hash"], r["source_hash"]) for r in sample}
    gold_mh = {mh for mh, _ in gold_pairs}
    with open(GOLD_OUT, "w") as f:
        for r in sample:
            f.write(json.dumps(r, default=str) + "\n")
    kept = []
    for s in stmts:
        mh = s.get_hash(refresh=True)
        if mh not in gold_mh:
            continue
        s.evidence = [ev for ev in (s.evidence or []) if (mh, ev.get_source_hash()) in gold_pairs]
        kept.append(s)
    json.dump(stmts_to_json(kept), open(STMTS_OUT, "w"))
    print(f"\nwrote {len(sample)} gold rows -> {GOLD_OUT}")
    print(f"wrote {len(kept)} statements ({sum(len(s.evidence or []) for s in kept)} ev to score) -> {STMTS_OUT}")

    # contamination guard
    try:
        import check_contamination as cc
        contam = cc.find_contamination(eval_paths=[GOLD_OUT])
        print(f"contamination guard: {'CLEAN ✓' if not contam else f'{len(contam)} OVERLAP(S)!'}")
    except Exception as e:  # noqa: BLE001
        print(f"contamination guard skipped: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
