"""Feasibility: can we bolster the eval set by pooling the WIDER curation universe
(all 48 curators), beyond the 2-curator INDRA assembly benchmark we already use?

Answers, with numbers, the five things that decide it:
  1. SIZE      — unique (statement, evidence) pairs in the universe vs how many are
                 NEW (not already in belief_benchmark, our existing 2-curator pool).
  2. RECOVER   — of the NEW pairs, how many carry ev_json (evidence text embedded
                 in the curation payload) — the only version-skew-proof way to get
                 evidence text for an arbitrary curator's pair.
  3. BALANCE   — gold correct/incorrect split of the recoverable-new pairs, hence
                 the balanced ceiling (2*min) a 1:1 eval could draw.
  4. CONTAM    — how many recoverable-new pairs collide with prior holdouts/evals/
                 fewshots (must be excluded).
  5. AGREEMENT — on pairs curated by >=2 curators, how often they agree — the
                 direct read on whether pooling strangers' labels is trustworthy.

Also dumps per-curator correct-rate (the "bias" made visible) and refreshes the
full-universe cache. Pass ``--offline-cache`` only for an explicitly offline run;
a stale cache is never silently treated as current.

    PYTHONPATH=src .venv/bin/python scripts/curation_pool_feasibility.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import importlib.util

from indra_belief.curation import aggregate_gold  # noqa: E402

_spec = importlib.util.spec_from_file_location("pmc", ROOT / "scripts" / "pull_my_curations.py")
pmc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pmc)

DATA = ROOT / "data" / "benchmark"
CACHE = ROOT / "data" / "results" / "curation_universe_all.jsonl"
PAIR_CURATORS = {"bachmanjohn@gmail.com", "ben.gyori@gmail.com", "bachmanjohn", "ben.gyori"}


def load_universe(*, use_cache: bool = False) -> list[dict]:
    if use_cache:
        if not CACHE.exists():
            raise SystemExit(f"--offline-cache requested but cache is missing: {CACHE}")
        print(f"OFFLINE: using cached universe {CACHE}", flush=True)
        return [json.loads(l) for l in CACHE.open() if l.strip()]
    url, key, _ = pmc._load_env()
    data = pmc.pull_all(url, key)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
    print(f"pulled + cached {len(data)} curations -> {CACHE}", flush=True)
    return data


def pair_of(d: dict):
    mh, sh = d.get("pa_hash"), d.get("source_hash")
    if mh is None or sh is None:
        return None
    try:
        return (int(mh), int(sh))
    except (TypeError, ValueError):
        return None


def load_pairs(path: Path) -> set:
    out = set()
    if not path.exists():
        return out
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "pa_hash" in r and "source_hash" in r:
            try:
                out.add((int(r["pa_hash"]), int(r["source_hash"])))
            except (TypeError, ValueError):
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline-cache", action="store_true",
                    help="use the existing cache without claiming it is current")
    args = ap.parse_args()
    data = load_universe(use_cache=args.offline_cache)
    print(f"\nTOTAL curations: {len(data)}", flush=True)

    # group curations by pair
    tags_by_pair = defaultdict(list)
    curators_by_pair = defaultdict(set)
    evjson_by_pair = defaultdict(bool)
    for d in data:
        k = pair_of(d)
        if k is None:
            continue
        tags_by_pair[k].append(d.get("tag", ""))
        if d.get("curator"):
            curators_by_pair[k].add(d["curator"])
        evjson_by_pair[k] = evjson_by_pair[k] or bool(d.get("ev_json"))

    all_pairs = set(tags_by_pair)
    gold = {k: aggregate_gold(t) for k, t in tags_by_pair.items()}
    print(f"unique (stmt,evidence) pairs: {len(all_pairs)}", flush=True)
    print(f"unique statements: {len({mh for mh, _ in all_pairs})}", flush=True)

    # ── 1. NEW vs already-in-benchmark ──────────────────────────────────────
    bench = load_pairs(DATA / "belief_benchmark.jsonl")
    new_pairs = all_pairs - bench
    print(f"\n[1] SIZE", flush=True)
    print(f"    pairs already in belief_benchmark (2-curator pool we use): {len(all_pairs & bench)}", flush=True)
    print(f"    NEW pairs (not in belief_benchmark): {len(new_pairs)}", flush=True)

    # ── 2. RECOVERABILITY of NEW pairs (ev_json embedded) ───────────────────
    new_recov = {k for k in new_pairs if evjson_by_pair[k]}
    print(f"\n[2] RECOVERABILITY (ev_json embedded in payload)", flush=True)
    print(f"    NEW pairs with ev_json: {len(new_recov)} / {len(new_pairs)} "
          f"({100*len(new_recov)/max(len(new_pairs),1):.0f}%)", flush=True)

    # ── 3. BALANCE of recoverable-new ───────────────────────────────────────
    nc = sum(1 for k in new_recov if gold[k] == "correct")
    ni = sum(1 for k in new_recov if gold[k] == "incorrect")
    print(f"\n[3] BALANCE of recoverable-NEW pairs", flush=True)
    print(f"    correct: {nc}   incorrect: {ni}   balanced ceiling (2*min): {2*min(nc, ni)}", flush=True)

    # ── 4. CONTAMINATION overlap (pair-level) ───────────────────────────────
    contam = set()
    per_file = {}
    for pat in ("holdout*.jsonl", "eval_set*.jsonl", "eval_curation_v1*.jsonl", "fewshot*.jsonl", "external_gold*.jsonl"):
        for p in sorted(DATA.glob(pat)):
            fp = load_pairs(p)
            hit = new_recov & fp
            if hit:
                per_file[p.name] = len(hit)
            contam |= fp
    clean_new_recov = new_recov - contam
    cc_nc = sum(1 for k in clean_new_recov if gold[k] == "correct")
    cc_ni = sum(1 for k in clean_new_recov if gold[k] == "incorrect")
    print(f"\n[4] CONTAMINATION (recoverable-NEW pairs colliding with prior sets)", flush=True)
    for name, n in sorted(per_file.items()):
        print(f"    {name:32s} {n}", flush=True)
    print(f"    clean recoverable-NEW pairs: {len(clean_new_recov)} "
          f"({cc_nc} correct / {cc_ni} incorrect; balanced ceiling {2*min(cc_nc, cc_ni)})", flush=True)

    # ── 5. INTER-CURATOR AGREEMENT (all pairs with >=2 curators) ────────────
    multi = {k for k in all_pairs if len(curators_by_pair[k]) >= 2}
    agree = 0
    for k in multi:
        verdicts = set()
        # per-curator aggregated verdict on this pair
        per_cur = defaultdict(list)
        for d in data:
            kk = pair_of(d)
            if kk == k and d.get("curator"):
                per_cur[d["curator"]].append(d.get("tag", ""))
        for cur, tg in per_cur.items():
            verdicts.add(aggregate_gold(tg))
        if len(verdicts) == 1:
            agree += 1
    print(f"\n[5] INTER-CURATOR AGREEMENT", flush=True)
    print(f"    pairs curated by >=2 curators: {len(multi)}", flush=True)
    if multi:
        print(f"    of those, all curators agree (correct vs incorrect): {agree} "
              f"({100*agree/len(multi):.0f}%)", flush=True)

    # ── per-curator correct-rate (the bias, visible) ────────────────────────
    by_cur_pairs = defaultdict(lambda: defaultdict(list))
    for d in data:
        k = pair_of(d)
        if k is None or not d.get("curator"):
            continue
        by_cur_pairs[d["curator"]][k].append(d.get("tag", ""))
    print(f"\n[per-curator] correct-rate over their unique pairs (top 20 by volume)", flush=True)
    rows = []
    for cur, pairs in by_cur_pairs.items():
        g = [aggregate_gold(t) for t in pairs.values()]
        c = sum(1 for x in g if x == "correct")
        i = sum(1 for x in g if x == "incorrect")
        rows.append((cur, len(pairs), c, i, 100 * c / max(c + i, 1)))
    rows.sort(key=lambda r: -r[1])
    print(f"    {'curator':40s} {'pairs':>6} {'corr':>5} {'inc':>5} {'corr%':>6}", flush=True)
    for cur, n, c, i, pct in rows[:20]:
        flag = "  <2-curator pool>" if cur in PAIR_CURATORS else ""
        print(f"    {cur:40s} {n:6d} {c:5d} {i:5d} {pct:5.0f}%{flag}", flush=True)


if __name__ == "__main__":
    main()
