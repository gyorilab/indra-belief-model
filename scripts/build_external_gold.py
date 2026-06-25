"""E4 — build an EXTERNAL human-curation gold set from third-party (non-pair)
INDRA curators.

Why this exists (vs build_curation_eval.py): build_curation_eval.py draws its gold
from belief_benchmark.jsonl, which is the SAME two expert curators (ben.gyori +
bachmanjohn — "the pair") who authored every prior holdout/eval. Every benchmark
we ship is therefore graded against the pair's judgement. This builds a gold set
sourced ENTIRELY from OTHER curators in the public INDRA curation DB — a held-out
human population — so model accuracy can be measured against curators the prior
gold never saw.

Feasibility (scripts/probe_curation_universe.py, data/results/curation_universe_probe.json):
no single non-pair curator clears n>=100 balanced (best is imke.ditters at
bal_ceiling 82). So we POOL all non-pair curators (the plan's documented
contingency). The keyed /curation/list payload carries ev_json (evidence text +
source_api + pmid) and pa_json (subject/object canonical names + stmt_type +
matches_hash) INLINE, so every row is materialized from the list payload alone —
no second by-hash fetch, no fragile local hash-join. Pooled ev_json-recoverable
balance: 109 correct / 518 incorrect -> balanced ceiling 218, ample for n>=100.

Design (mirrors build_curation_eval.py exactly where it can):
- PAIR EXCLUSION: ben.gyori / bachmanjohn curations are dropped before gold is
  aggregated, so a pair "correct" can never out-vote a third curator. Gold is the
  any-incorrect-wins aggregate (curation.aggregate_gold) over NON-PAIR tags only.
- FRESH POOL: exclude every (pa_hash, source_hash) pair appearing in ANY prior
  gold/holdout/eval/fewshot file (EXCLUDE_GLOBS), so external_gold is disjoint
  from all existing gold by construction.
- FEWSHOT-UNIVERSE EXCLUSION: drop pairs whose evidence/entity-pair the model
  sees at inference (reuse check_contamination.load_all_examples), identical
  policy to build_curation_eval.is_contaminated.
- FORCED 1:1, STRATIFIED by stmt_type (reuse the same stratified_balanced logic):
  per type take min(#correct,#incorrect) of each class; the larger class is
  sampled to mirror the smaller class's source_api mix.
- Output is the flat belief_benchmark/holdout row (drop-in for the scorer + the
  contamination guard), plus gold/all_tags/curators for provenance.
- SELF-GUARD: assert cc.find_contamination([OUT]) == [] (node A1 repaired guard),
  exactly as build_curation_eval.py self-guards. Fail loudly otherwise.

    PYTHONPATH=src python scripts/build_external_gold.py
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import check_contamination as cc  # noqa: E402 — single source of truth for the fewshot universe
from build_curation_eval import (  # noqa: E402 — reuse, don't duplicate
    fewshot_filters,
    is_contaminated,
    stratified_balanced,
)
from indra_belief.curation import aggregate_gold, is_gold_correct  # noqa: E402

DATA = ROOT / "data" / "benchmark"
OUT = DATA / "external_gold_v1.jsonl"
SEED = 20260623  # date-less seed per the brief; fixed for reproducibility

# The two pair curators whose judgement backs ALL prior gold. Excluded so the
# external gold is a genuinely held-out human population.
PAIR = {"ben.gyori@gmail.com", "bachmanjohn@gmail.com", "ben.gyori", "bachmanjohn"}

# Every file whose pairs must NOT appear in the external gold: prior holdouts,
# eval sets, fewshot pools, the pair-curator benchmark itself, the rasmachine
# gold, probe sets, and calibration sets. Globbed so any future sibling is
# excluded automatically. (build_curation_eval used only holdout/eval/fewshot;
# we additionally exclude belief_benchmark + rasmachine + probe + calibration so
# external_gold is disjoint from EVERY existing gold artifact, not just evals.)
EXCLUDE_GLOBS = (
    "holdout*.jsonl",
    "eval_set*.jsonl",
    "eval_curation*.jsonl",
    "fewshot*.jsonl",
    "belief_benchmark.jsonl",
    "rasmachine*.jsonl",
    "probe_*.jsonl",
    "calibration_*.jsonl",
)


def _load_env() -> tuple[str, str]:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return (os.environ["INDRA_DB_REST_URL"].rstrip("/"),
            os.environ["INDRA_DB_REST_API_KEY"])


def pull_all(url: str, key: str) -> list[dict]:
    import httpx
    last = None
    for attempt in range(8):
        try:
            with httpx.Client(headers={"User-Agent": "indra-belief-e4/1"},
                              timeout=httpx.Timeout(240.0, connect=30.0)) as c:
                r = c.get(f"{url}/curation/list", params={"api_key": key})
                r.raise_for_status()
                data = r.json()
                if isinstance(data, dict):
                    raise SystemExit(f"list-all returned error dict: {data}")
                return data
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"  pull attempt {attempt} failed: {type(e).__name__}: {e}", flush=True)
    raise SystemExit(f"list-all failed all retries: {last}")


def load_exclude_pairs() -> tuple[set[tuple[int, int]], dict[str, int]]:
    """Every (pa_hash, source_hash) pair in any prior gold/eval/holdout/fewshot
    file — the external gold must share none of them."""
    pairs: set[tuple[int, int]] = set()
    per_file: dict[str, int] = {}
    seen: set[Path] = set()
    for pat in EXCLUDE_GLOBS:
        for path in sorted(DATA.glob(pat)):
            if path == OUT or path in seen:
                continue
            seen.add(path)
            fp: set[tuple[int, int]] = set()
            for line in open(path):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "pa_hash" in r and "source_hash" in r:
                    try:
                        fp.add((int(r["pa_hash"]), int(r["source_hash"])))
                    except (TypeError, ValueError):
                        pass
            per_file[path.name] = len(fp)
            pairs |= fp
    return pairs, per_file


def _ev_pa(d: dict) -> tuple[dict | None, dict | None]:
    """Parse a curation row's inline ev_json / pa_json (each may be a JSON str)."""
    def _j(x):
        if x is None:
            return None
        if isinstance(x, str):
            try:
                return json.loads(x)
            except json.JSONDecodeError:
                return None
        return x
    return _j(d.get("ev_json")), _j(d.get("pa_json"))


def _name(agent) -> str:
    return ((agent or {}).get("name") or "").strip() if isinstance(agent, dict) else ""


def _subj_obj(pa: dict) -> tuple[str, str]:
    """Map an INDRA statement's agents to (subject, object) the SAME way
    belief_benchmark does, across every statement schema the curation DB emits:

      - subj/obj          (Activation, Inhibition, RegulateAmount, ...)
      - enz/sub           (Phosphorylation, Sumoylation, Acetylation, modifications)
      - members[0/1]      (Complex)

    Single-agent statements (ActiveForm, Autophosphorylation, Translocation,
    Glycosylation) have no second agent -> returns ("","") so the caller drops
    them, exactly as build_curation_eval skips subject=="?" and object=="?".
    """
    subj = _name(pa.get("subj"))
    obj = _name(pa.get("obj"))
    if subj and obj:
        return subj, obj
    enz, sub = _name(pa.get("enz")), _name(pa.get("sub"))
    if enz and sub:
        return enz, sub
    members = pa.get("members")
    if isinstance(members, list) and len(members) >= 2:
        m0, m1 = _name(members[0]), _name(members[1])
        if m0 and m1:
            return m0, m1
    return "", ""


def aggregate_external_pairs(data: list[dict]) -> dict[tuple[int, int], dict]:
    """Collapse non-pair curations to one record per (pa_hash, source_hash).

    Gold = any-incorrect-wins over NON-PAIR tags only (pair curations are dropped
    before aggregation). Evidence text + entities come from the inline ev_json /
    pa_json the keyed list payload carries — no second fetch. A pair is kept only
    if at least one of its non-pair curations carries ev_json (so we can
    materialize evidence_text) and a parseable pa_json (subject/object/type)."""
    tags_by_pair: dict[tuple[int, int], list[str]] = defaultdict(list)
    curators_by_pair: dict[tuple[int, int], list[str]] = defaultdict(list)
    rep_by_pair: dict[tuple[int, int], dict] = {}

    for d in data:
        if d.get("curator") in PAIR:
            continue  # strict pair exclusion: never let the pair vote
        mh, sh = d.get("pa_hash"), d.get("source_hash")
        if mh is None or sh is None:
            continue
        try:
            key = (int(mh), int(sh))
        except (TypeError, ValueError):
            continue
        ev, pa = _ev_pa(d)
        tags_by_pair[key].append(str(d.get("tag", "")))
        cur = str(d.get("curator", "") or "")
        if cur and cur not in curators_by_pair[key]:
            curators_by_pair[key].append(cur)
        # Build the representative flat row the first time we can fully materialize
        # it (need ev text + subj/obj/type). Prefer a row that has both.
        if key not in rep_by_pair and ev and pa and isinstance(pa, dict):
            text = (ev.get("text") or "").strip()
            subj, obj = _subj_obj(pa)
            stmt_type = (pa.get("type") or "").strip()
            if not text or not subj or not obj or not stmt_type:
                continue
            text_refs = ev.get("text_refs") or {}
            pmid = ev.get("pmid") or text_refs.get("PMID") or text_refs.get("pmid") or ""
            rep_by_pair[key] = {
                "pa_hash": key[0],
                "source_hash": key[1],
                "matches_hash": str(key[0]),
                "stmt_type": stmt_type,
                "subject": subj,
                "object": obj,
                "evidence_text": text,
                "source_api": ev.get("source_api") or "",
                "pmid": str(pmid) if pmid else "",
            }

    out: dict[tuple[int, int], dict] = {}
    for key, rep in rep_by_pair.items():
        tags = tags_by_pair[key]
        gold = aggregate_gold(tags)
        if gold is None:
            continue
        rep = dict(rep)
        rep["gold"] = gold
        rep["all_tags"] = sorted(set(tags))
        rep["curators"] = curators_by_pair[key]
        # gold-aligned `tag` so record.tag=="correct" matches the gold rule
        if gold == "correct":
            rep["tag"] = "correct"
        else:
            rep["tag"] = next((t for t in tags if not is_gold_correct(t)),
                              tags[0] if tags else "other")
        out[key] = rep
    return out


def main(seed: int = SEED) -> None:
    rng = random.Random(seed)
    url, key = _load_env()
    print("pulling /curation/list (keyed list-all) ...", flush=True)
    data = pull_all(url, key)
    print(f"TOTAL curations pulled: {len(data)}")

    pairs = aggregate_external_pairs(data)
    print(f"non-pair pairs materialized from ev_json+pa_json: {len(pairs)}")
    nc0 = sum(1 for r in pairs.values() if r["gold"] == "correct")
    ni0 = sum(1 for r in pairs.values() if r["gold"] == "incorrect")
    print(f"  gold balance (pre-exclusion): {nc0} correct / {ni0} incorrect")

    exclude, per_file = load_exclude_pairs()
    print(f"\nexclude pairs (prior gold/holdout/eval/fewshot): {len(exclude)}")
    for name, n in sorted(per_file.items()):
        print(f"    {name:34s} {n}")

    fresh = {k: r for k, r in pairs.items() if k not in exclude}
    dropped_overlap = len(pairs) - len(fresh)
    print(f"\npair-overlap dropped: {dropped_overlap}")

    # fewshot-universe exclusion (reuse build_curation_eval helpers verbatim)
    ex_pairs, ex_norms = fewshot_filters()
    before = len(fresh)
    fresh = {k: r for k, r in fresh.items()
             if not is_contaminated(r, ex_pairs, ex_norms)}
    dropped_fewshot = before - len(fresh)
    print(f"fewshot-universe dropped: {dropped_fewshot} "
          f"({len(ex_pairs)} entity pairs, {len(ex_norms)} evidence strings)")

    fresh_rows = list(fresh.values())
    nc = sum(1 for r in fresh_rows if r["gold"] == "correct")
    ni = sum(1 for r in fresh_rows if r["gold"] == "incorrect")
    print(f"\nFRESH pool (de-contaminated): {len(fresh_rows)} pairs "
          f"({nc} correct / {ni} incorrect)")

    sample, report = stratified_balanced(fresh_rows, rng)
    rng.shuffle(sample)
    s_nc = sum(1 for r in sample if r["gold"] == "correct")
    s_ni = sum(1 for r in sample if r["gold"] == "incorrect")
    print(f"\nEXTERNAL GOLD: {len(sample)} pairs ({s_ni} incorrect / {s_nc} correct)")
    print(f"stratified 1:1 by stmt_type: {report['types_used']} types used, "
          f"dropped (single-class): {report['types_dropped']}")

    print("\nstmt_type x class (incorrect | correct) — identical by construction:")
    by_type = defaultdict(lambda: [0, 0])
    for r in sample:
        by_type[r["stmt_type"]][0 if r["gold"] == "incorrect" else 1] += 1
    for t, (i_n, c_n) in sorted(by_type.items(), key=lambda kv: -sum(kv[1])):
        print(f"    {t:20s} {i_n:4d} | {c_n:4d}")

    print("\ncontributing curators (rows in final set):")
    cur_n = Counter()
    for r in sample:
        for c in r["curators"]:
            cur_n[c] += 1
    for c, n in cur_n.most_common():
        print(f"    {c:40s} {n}")

    with open(OUT, "w") as f:
        for r in sample:
            f.write(json.dumps(r) + "\n")

    meta = {
        "seed": seed,
        "source": "db.indra.bio /curation/list (keyed list-all)",
        "output": str(OUT.relative_to(ROOT)),
        "n": len(sample),
        "n_incorrect": s_ni,
        "n_correct": s_nc,
        "balance": "1:1 forced",
        "matching": "stratified 1:1 by stmt_type; source_api matched within type",
        "gold_rule": "any-incorrect-wins (curation.aggregate_gold) over NON-PAIR tags only",
        "pair_excluded": sorted(PAIR),
        "n_curators_contributing": len(cur_n),
        "curators_contributing": dict(cur_n.most_common()),
        "fresh_pool_size": len(fresh_rows),
        "fresh_pool_correct": nc,
        "fresh_pool_incorrect": ni,
        "materialized_pairs": len(pairs),
        "dropped_pair_overlap": dropped_overlap,
        "dropped_fewshot_universe": dropped_fewshot,
        "exclude_globs": list(EXCLUDE_GLOBS),
        "exclude_pairs_total": len(exclude),
        "exclude_per_file": per_file,
        "stratification": report,
        "evidence_source": "inline ev_json (text/source_api/pmid) + pa_json (subj/obj/type) from the list payload; no second fetch",
    }

    # Self-guard (node A1 repaired contamination checker) — fail loudly on leak.
    contam = cc.find_contamination(eval_paths=[OUT])
    meta["contamination_check"] = {
        "checker": "scripts/check_contamination.py (find_contamination)",
        "overlaps": len(contam),
        "result": "CLEAN" if not contam else "CONTAMINATED",
    }
    meta_path = OUT.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nSaved {OUT}")
    print(f"Saved {meta_path}")
    if contam:
        raise SystemExit(
            f"CONTAMINATION in {OUT.name}: {len(contam)} overlap(s) survived — "
            f"first: [{contam[0]['kind']}] {contam[0].get('source')}"
        )
    print("contamination guard: CLEAN")


if __name__ == "__main__":
    main()
