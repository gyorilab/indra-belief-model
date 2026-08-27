"""Build a balanced, zero-leakage human-curation eval set from the INDRA
assembly benchmark.

Why this exists (vs build_holdout.py): build_holdout.py freezes the canonical
200-record holdout (seed 42, partial exclusions, natural tag mix). This builds a
DIFFERENT artifact for the MedPsy-vs-gemma head-to-head — the rasmachine gold was
12% negative and version-skewed, useless for error-detection. The INDRA assembly
benchmark (data/benchmark/belief_benchmark.jsonl) is the real human-curation
universe: 5,709 curated (statement, evidence) pairs, 40% negative, two expert
curators, evidence + curations from the SAME snapshot (so source_hash joins by
construction — no version skew).

Design decisions (chosen with the user):
- FRESH POOL only: exclude every (pa_hash, source_hash) pair that appears in ANY
  prior holdout/eval/few-shot pool. Guarantees zero leakage from past evals.
- FORCED 1:1 balance: equal correct/incorrect. Accuracy on a 40%-neg set is still
  dominated by the easy positive class; 1:1 makes error-detection the headline.
- STRATIFIED 1:1 by stmt_type, with source_api matched WITHIN each type: for each
  statement type take min(#correct, #incorrect) of each class, so the stmt_type
  marginal is IDENTICAL across classes (Complex vs Phosphorylation difficulty
  can't masquerade as a label effect). Within a type, the larger class is sampled
  to mirror the smaller class's reader (source_api) mix. A joint stmt_type×reader
  exact match was rejected: it costs ~30% of the pairs and drops single-class
  reader×type cells (itself a selection bias). stmt_type is the dominant confound;
  reader is balanced secondarily and its residual marginal is reported.

Gold rule: any-incorrect-wins per pair (curation.aggregate_gold), so a pair
curated both 'correct' and 'wrong_relation' is incorrect.

Output schema is the flat belief_benchmark/holdout row (drop-in for
indra_belief.data.corpus.CorpusIndex.build_records and the monolithic scorer),
plus `gold` and `all_tags` for provenance. A sidecar .meta.json records the
seed, exclusions, and stratum-match quality.

    PYTHONPATH=src python scripts/build_curation_eval.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import check_contamination as cc  # noqa: E402 — single source of truth for the fewshot universe
from indra_belief.curation import aggregate_gold, is_gold_correct  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gold_output import add_rebuild_flag, guard_outputs  # noqa: E402

DATA = ROOT / "data" / "benchmark"
BENCHMARK = DATA / "belief_benchmark.jsonl"
OUT = DATA / "eval_curation_v1.jsonl"
SEED = 20260607

# Every file whose pairs must NOT appear in the fresh eval set: prior holdouts,
# prior eval sets, and the few-shot pools the model sees at inference. Globbed so
# any future holdout_*/eval_set_*/fewshot_* is excluded automatically.
LEAKAGE_GLOBS = ("holdout*.jsonl", "eval_set*.jsonl", "fewshot*.jsonl")


def _pair(row: dict) -> tuple[int, int]:
    return (int(row["pa_hash"]), int(row["source_hash"]))


def fewshot_filters() -> tuple[set[tuple[str, str]], set[str]]:
    """The full inference-time fewshot universe (example_bank + probe shots +
    inline prompt examples + unified curriculum), reduced to the two things we
    exclude on: (subject, object) pairs and normalized evidence strings. Reuses
    check_contamination so this builder and the guard never disagree."""
    ex_pairs: set[tuple[str, str]] = set()
    ex_norms: set[str] = set()
    for ex in cc.load_all_examples():
        ev = ex.get("evidence", "") or ""
        if ev:
            n = cc._norm(ev)
            if n:
                ex_norms.add(n)
        if ex.get("claim"):
            s, o = cc._parse_legacy_claim(ex["claim"])
            if s and o:
                ex_pairs.add((s, o))
    return ex_pairs, ex_norms


def is_contaminated(row: dict, ex_pairs: set, ex_norms: set) -> bool:
    """Mirror check_contamination's match policy (exact / substring / 50-char
    paraphrase / entity-pair). The post-build assert is the real guarantee;
    this just removes hits up front so the balance survives."""
    if (row.get("subject"), row.get("object")) in ex_pairs:
        return True
    n = cc._norm(row.get("evidence_text", "") or "")
    if not n:
        return False
    if n in ex_norms:
        return True
    if len(n) >= 30:
        for en in ex_norms:
            if len(en) >= 30 and (en in n or n in en):
                return True
    if len(n) >= 50:
        for en in ex_norms:
            if len(en) < 50:
                continue
            for i in range(0, len(en) - 50 + 1, 5):
                if en[i:i + 50] in n:
                    return True
    return False


def load_leakage_pairs() -> tuple[set[tuple[int, int]], dict[str, int]]:
    pairs: set[tuple[int, int]] = set()
    per_file: dict[str, int] = {}
    seen_files: set[Path] = set()
    for pat in LEAKAGE_GLOBS:
        for path in sorted(DATA.glob(pat)):
            if path == OUT or path in seen_files:
                continue
            seen_files.add(path)
            fp: set[tuple[int, int]] = set()
            for line in open(path):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if "pa_hash" in r and "source_hash" in r:
                    fp.add(_pair(r))
            per_file[path.name] = len(fp)
            pairs |= fp
    return pairs, per_file


def aggregate_pairs() -> dict[tuple[int, int], dict]:
    """Collapse belief_benchmark's multi-curation rows to one record per pair,
    with any-incorrect-wins gold and the full tag list preserved."""
    by_pair: dict[tuple[int, int], dict] = {}
    tags_by_pair: dict[tuple[int, int], list[str]] = defaultdict(list)
    for line in open(BENCHMARK):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if not r.get("evidence_text"):
            continue
        if r.get("subject") == "?" and r.get("object") == "?":
            continue  # unevaluable
        key = _pair(r)
        tags_by_pair[key].append(r["tag"])
        # keep the first row as the representative metadata carrier
        by_pair.setdefault(key, r)
    # finalize gold + provenance
    out: dict[tuple[int, int], dict] = {}
    for key, rep in by_pair.items():
        tags = tags_by_pair[key]
        gold = aggregate_gold(tags)
        rep = dict(rep)
        rep["gold"] = gold
        rep["all_tags"] = sorted(set(tags))
        # make `tag` gold-aligned so record.tag=="correct" matches the gold rule:
        # keep 'correct' iff gold is correct, else the first dissenting tag.
        if gold == "correct":
            rep["tag"] = "correct"
        else:
            rep["tag"] = next((t for t in tags if not is_gold_correct(t)), rep["tag"])
        out[key] = rep
    return out


def _covariate_pick(
    pool: list[dict], target: list[dict], keyfn, n: int, rng: random.Random
) -> list[dict]:
    """Pick `n` rows from `pool` whose `keyfn` distribution mirrors `target`'s.
    Per key bucket take min(needed, available); fill any shortfall from leftover
    (shuffled). Returns exactly min(n, len(pool)) rows."""
    need = Counter(keyfn(r) for r in target)
    buckets: dict[object, list[dict]] = defaultdict(list)
    for r in pool:
        buckets[keyfn(r)].append(r)
    for b in buckets:
        rng.shuffle(buckets[b])
    picked: list[dict] = []
    for b, k in need.items():
        avail = buckets.get(b, [])
        take = min(k, len(avail))
        picked.extend(avail[:take])
        buckets[b] = avail[take:]
    if len(picked) < n:
        leftover = [r for b in buckets for r in buckets[b]]
        rng.shuffle(leftover)
        picked.extend(leftover[: n - len(picked)])
    return picked[:n]


def stratified_balanced(fresh: list[dict], rng: random.Random) -> tuple[list[dict], dict]:
    """Stratified 1:1 by stmt_type: per type take min(#correct,#incorrect) of
    EACH class (so the stmt_type marginal is identical across classes), with the
    larger class sampled to mirror the smaller class's source_api mix. Types with
    only one class present contribute nothing (can't form a within-type balance)."""
    by_type: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"correct": [], "incorrect": []})
    for r in fresh:
        by_type[r.get("stmt_type") or "?"][r["gold"]].append(r)

    picked: list[dict] = []
    per_type: dict[str, int] = {}
    dropped: list[str] = []
    for t, cls in by_type.items():
        C, I = cls["correct"], cls["incorrect"]
        m = min(len(C), len(I))
        if m == 0:
            dropped.append(t)
            continue
        small, large = (I, C) if len(I) <= len(C) else (C, I)
        rng.shuffle(small)
        small_sel = small[:m]
        large_sel = _covariate_pick(large, small_sel, lambda r: r.get("source_api") or "?", m, rng)
        picked.extend(small_sel)
        picked.extend(large_sel)
        per_type[t] = m

    report = {
        "types_used": len(per_type),
        "types_dropped": dropped,
        "per_type_pairs_each_class": dict(sorted(per_type.items(), key=lambda kv: -kv[1])),
    }
    return picked, report


def main(seed: int = SEED) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0])
    add_rebuild_flag(parser)
    guard_outputs([OUT, OUT.with_suffix('.meta.json')], rebuild=parser.parse_args().rebuild)

    rng = random.Random(seed)

    leakage, per_file = load_leakage_pairs()
    pairs = aggregate_pairs()
    print(f"belief_benchmark evaluable pairs: {len(pairs)}")
    print(f"leakage pairs (prior holdouts/evals/fewshots): {len(leakage)}")
    for name, n in sorted(per_file.items()):
        print(f"    {name:28s} {n}")

    fresh_rows = [r for k, r in pairs.items() if k not in leakage]

    # Exclude the inference-time fewshot universe (example_bank, probe shots,
    # inline prompt examples) — the model SEES these, so evaluating on them leaks.
    ex_pairs, ex_norms = fewshot_filters()
    before = len(fresh_rows)
    fresh_rows = [r for r in fresh_rows if not is_contaminated(r, ex_pairs, ex_norms)]
    print(f"fewshot-universe exclusion: {before - len(fresh_rows)} pairs dropped "
          f"({len(ex_pairs)} entity pairs, {len(ex_norms)} evidence strings)")

    correct = [r for r in fresh_rows if r["gold"] == "correct"]
    incorrect = [r for r in fresh_rows if r["gold"] == "incorrect"]
    print(f"\nFRESH pool (de-contaminated): {len(fresh_rows)} pairs  "
          f"({len(correct)} correct / {len(incorrect)} incorrect)")

    sample, report = stratified_balanced(fresh_rows, rng)
    rng.shuffle(sample)

    ni = sum(1 for r in sample if r["gold"] == "incorrect")
    nc = sum(1 for r in sample if r["gold"] == "correct")
    print(f"\nEVAL SET: {len(sample)} pairs  ({ni} incorrect / {nc} correct)")
    print(f"stratified 1:1 by stmt_type: {report['types_used']} types used, "
          f"dropped (single-class): {report['types_dropped']}")

    print("\ntag distribution (gold-aligned):")
    for t, c in Counter(r["tag"] for r in sample).most_common():
        print(f"    {t:20s} {c}")
    print("\nstmt_type × class (incorrect | correct) — identical by construction:")
    by_type = defaultdict(lambda: [0, 0])
    for r in sample:
        by_type[r["stmt_type"]][0 if r["gold"] == "incorrect" else 1] += 1
    for t, (i_n, c_n) in sorted(by_type.items(), key=lambda kv: -sum(kv[1])):
        print(f"    {t:20s} {i_n:4d} | {c_n:4d}")
    print("\nsource_api × class (incorrect | correct) — matched within type, residual:")
    by_api = defaultdict(lambda: [0, 0])
    for r in sample:
        by_api[r.get("source_api") or "?"][0 if r["gold"] == "incorrect" else 1] += 1
    for a, (i_n, c_n) in sorted(by_api.items(), key=lambda kv: -sum(kv[1])):
        print(f"    {a:20s} {i_n:4d} | {c_n:4d}")

    with open(OUT, "w") as f:
        for r in sample:
            f.write(json.dumps(r) + "\n")

    meta = {
        "seed": seed,
        "source": str(BENCHMARK.relative_to(ROOT)),
        "output": str(OUT.relative_to(ROOT)),
        "n": len(sample),
        "n_incorrect": sum(1 for r in sample if r["gold"] == "incorrect"),
        "n_correct": sum(1 for r in sample if r["gold"] == "correct"),
        "fresh_pool_size": len(fresh_rows),
        "leakage_pairs_excluded": len(leakage),
        "leakage_per_file": per_file,
        "stratification": report,
        "balance": "1:1 forced",
        "matching": "stratified 1:1 by stmt_type; source_api matched within type",
        "gold_rule": "any-incorrect-wins (curation.aggregate_gold)",
    }
    meta_path = OUT.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nSaved {OUT}")
    print(f"Saved {meta_path}")

    # Self-guard: the output MUST be contamination-free against the same checker
    # CI runs. Fail loudly rather than emit a leaky eval set.
    contam = cc.find_contamination(eval_paths=[OUT])
    if contam:
        raise SystemExit(
            f"CONTAMINATION in {OUT.name}: {len(contam)} overlap(s) survived — "
            f"first: [{contam[0]['kind']}] {contam[0]['source']}"
        )
    print("contamination guard: CLEAN ✓")


if __name__ == "__main__":
    main()
