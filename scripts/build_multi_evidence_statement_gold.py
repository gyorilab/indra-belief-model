#!/usr/bin/env python3
"""Build multi-evidence STATEMENT gold sets to validate statement-level belief.

Every eval corpus we have is ~100% single-evidence per statement, so
``statement_belief.py``'s per-evidence -> statement aggregation (source-aware
hybrid log-odds) has never been exercised. This assembles two complementary
multi-evidence statement golds from data already on disk (no network):

  TIER 1 -- ``eval_curation_v1`` re-grained. 342 statements that each carry >=2
    DISTINCT human-curated evidences (single curator). Statement truth is
    DERIVED from multiple human judgments via the noisy-OR rollup below, so the
    aggregation can actually be falsified. 39 statements have MIXED per-evidence
    gold -- the discriminating cases. CAVEAT: in-sample (same rows back the
    shipped n=1606 evidence-grain eval) -> an aggregation-correctness check, not
    held-out generalization.

  TIER 2 -- representative-403 (curated via /curate by one curator) joined to the
    frozen full-evidence substrate. 335/403 statements gain their real evidence
    body (median ~17 evidences). Held-out and de-contaminated (verified disjoint
    from Tier 1). Only one evidence per statement is human-labeled, so statement
    truth is the single curation TRANSFERRED to the statement -- the multiplicity
    stress set, weaker truth than Tier 1.

The one new modelling choice is ``statement_gold_rollup``: INDRA membership
semantics -- a statement is correctly extracted iff at least ONE of its curated
evidences supports it. This is any-CORRECT-wins, the opposite of
``curation.aggregate_gold`` (any-incorrect-wins over tags on a single evidence),
so that function is deliberately NOT reused.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_CV1 = ROOT / "data/benchmark/eval_curation_v1.jsonl"
REP403_GOLD = ROOT / "data/benchmark/representative_indra_curations_400.jsonl"
SUBSTRATE = ROOT / "data/corpora/representative_indra_403_full_evidence/.frozen-substrate.work"
OUT_DIR = ROOT / "data/benchmark"


def statement_gold_rollup(per_evidence_golds) -> str | None:
    """Roll per-evidence human curations up to one statement-level truth.

    INDRA membership semantics: a statement is a correct extraction iff at least
    one of its curated evidences genuinely supports it. So the statement is
    ``correct`` if ANY curated evidence is correct, ``incorrect`` only if ALL are
    incorrect, and ``None`` if nothing was curated. This is any-CORRECT-wins and
    mirrors ``statement_belief``'s gate (an evidence is 'included' iff its verdict
    is correct); it is NOT ``curation.aggregate_gold`` (any-incorrect over tags).
    """
    golds = [g for g in per_evidence_golds if g in ("correct", "incorrect")]
    if not golds:
        return None
    return "correct" if any(g == "correct" for g in golds) else "incorrect"


def _claim(subject, stmt_type, obj) -> str:
    return f"{subject} [{stmt_type}] {obj}"


def _agent_name(a) -> str:
    return a.get("name") if isinstance(a, dict) else str(a)


def _render_stmt(stmt) -> str:
    """Best-effort readable claim from an INDRA statement JSON dict."""
    if not isinstance(stmt, dict):
        return str(stmt)
    t = stmt.get("type", "?")
    if isinstance(stmt.get("members"), list):
        return f"{' + '.join(_agent_name(m) for m in stmt['members'])} [{t}]"
    subj = stmt.get("subj") or stmt.get("enz")
    obj = stmt.get("obj") or stmt.get("sub")
    if subj is not None and obj is not None:
        return f"{_agent_name(subj)} [{t}] {_agent_name(obj)}"
    for k in ("agent", "sub", "obj", "subj"):
        if stmt.get(k) is not None:
            return f"{_agent_name(stmt[k])} [{t}]"
    return f"[{t}] {stmt.get('matches_hash', '')}"


def build_tier1() -> list[dict]:
    """eval_curation_v1 -> statements with >=2 distinct human-labeled evidences."""
    rows = [json.loads(l) for l in EVAL_CV1.open() if l.strip()]
    by_stmt: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_stmt[str(r["matches_hash"])].append(r)

    out = []
    for mh, rs in by_stmt.items():
        # one measurement per distinct evidence (source_hash); if a source_hash
        # somehow repeats, any-incorrect-wins picks the conservative per-evidence label.
        by_ev: dict[str, dict] = {}
        for r in rs:
            sh = str(r["source_hash"])
            prev = by_ev.get(sh)
            if prev is None or (r.get("gold") == "incorrect" and prev.get("gold") != "incorrect"):
                by_ev[sh] = r
        if len(by_ev) < 2:
            continue
        evs = list(by_ev.values())
        per_gold = [e.get("gold") for e in evs]
        stmt_gold = statement_gold_rollup(per_gold)
        r0 = evs[0]
        out.append({
            "matches_hash": mh,
            "tier": "evalcv1",
            "claim": _claim(r0.get("subject"), r0.get("stmt_type"), r0.get("object")),
            "subject": r0.get("subject"), "object": r0.get("object"),
            "stmt_type": r0.get("stmt_type"), "pmid": r0.get("pmid"),
            "statement_gold": stmt_gold,
            "gold_rule": "any-correct-wins (noisy-OR membership)",
            "mixed_gold": len({g for g in per_gold if g in ("correct", "incorrect")}) >= 2,
            "n_curated_evidence": len(evs),
            "n_evidence": len(evs),
            "curators": sorted({e.get("curator") for e in evs if e.get("curator")}),
            "evidences": [{
                "source_hash": str(e["source_hash"]),
                "source_api": e.get("source_api"),
                "evidence_text": e.get("evidence_text"),
                "gold": e.get("gold"),
                "tag": e.get("tag"),
                "curated": True,
            } for e in evs],
        })
    return out


def _ev_text(entry: dict) -> str | None:
    ev = entry.get("evidence")
    if isinstance(ev, dict) and ev.get("text"):
        return ev["text"]
    raw = entry.get("raw_evidence_json")
    if isinstance(raw, str):
        try:
            return json.loads(raw).get("text")
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def build_tier2(cap: int = 25) -> list[dict]:
    """representative-403 curation joined to the frozen full-evidence substrate.

    Each statement keeps its human-curated evidence plus up to ``cap`` others
    (deterministic by source_hash), so the heavy tail (max ~30k evidences) does
    not make scoring unbounded. Only the curated evidence is human-labeled.
    """
    gold_rows = [json.loads(l) for l in REP403_GOLD.open() if l.strip()]
    gold_by_mh = {str(r["matches_hash"]): r for r in gold_rows}

    # stream the 370MB substrate; keep a bounded candidate list per statement.
    keep = max(cap * 3, 80)
    cand: dict[str, dict[str, dict]] = defaultdict(dict)  # mh -> {source_hash: {text, api}}
    with (SUBSTRATE / "evidence_entries.jsonl").open() as fh:
        for line in fh:
            if not line.strip():
                continue
            e = json.loads(line)
            mh = str(e.get("matches_hash"))
            if mh not in gold_by_mh:
                continue
            sh = str(e.get("source_hash"))
            d = cand[mh]
            if sh in d or len(d) < keep:
                text = _ev_text(e)
                if text:
                    d[sh] = {"source_hash": sh, "source_api": e.get("source_api"), "evidence_text": text}

    out = []
    for mh, gr in gold_by_mh.items():
        pool = cand.get(mh, {})
        curated_sh = str(gr.get("source_hash"))
        # Seed the curated evidence FIRST (so it always survives and counts toward
        # the cap), synthesizing its text from the gold row if the substrate lacked
        # it; then fill deterministically with others up to the cap.
        chosen: dict[str, dict] = {curated_sh: pool.get(curated_sh, {
            "source_hash": curated_sh, "source_api": gr.get("source_api"),
            "evidence_text": gr.get("evidence_text"),
        })}
        for sh in sorted(pool):
            if len(chosen) >= cap:
                break
            chosen.setdefault(sh, pool[sh])
        evs = []
        for sh, ev in chosen.items():
            is_curated = sh == curated_sh
            evs.append({
                "source_hash": sh, "source_api": ev.get("source_api"),
                "evidence_text": ev.get("evidence_text"),
                "gold": gr.get("gold") if is_curated else None,
                "tag": gr.get("tag") if is_curated else None,
                "curated": is_curated,
            })
        out.append({
            "matches_hash": mh,
            "tier": "representative403",
            "claim": _render_stmt(gr.get("statement")),
            "stmt_type": gr.get("stmt_type"), "pmid": gr.get("pmid"),
            "statement_gold": gr.get("gold"),
            "gold_rule": "single-curation-transfer",
            "mixed_gold": False,
            "n_curated_evidence": 1,
            "n_evidence": len(evs),
            "curators": [gr.get("curator")] if gr.get("curator") else [],
            "evidences": evs,
        })
    return out


def _summary(name, recs):
    from collections import Counter
    mult = Counter(r["n_evidence"] for r in recs)
    gold = Counter(r["statement_gold"] for r in recs)
    mixed = sum(1 for r in recs if r["mixed_gold"])
    total_ev = sum(r["n_evidence"] for r in recs)
    print(f"[{name}] {len(recs)} statements | {total_ev} evidences "
          f"(mean {total_ev/max(1,len(recs)):.1f}/stmt) | gold {dict(gold)} | mixed {mixed}")
    print(f"         evidences/stmt: {dict(sorted(mult.items()))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=25, help="max evidences per Tier-2 statement")
    ap.add_argument("--tier2", action="store_true", help="also build Tier 2 (reads the 370MB substrate)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    t1 = build_tier1()
    p1 = OUT_DIR / "multi_evidence_statement_gold_evalcv1.jsonl"
    p1.write_text("".join(json.dumps(r) + "\n" for r in t1))
    _summary("TIER1 evalcv1", t1)
    print(f"  -> {p1}")

    if args.tier2:
        t2 = build_tier2(cap=args.cap)
        p2 = OUT_DIR / "multi_evidence_statement_gold_representative403.jsonl"
        p2.write_text("".join(json.dumps(r) + "\n" for r in t2))
        _summary("TIER2 representative403", t2)
        print(f"  -> {p2}")


if __name__ == "__main__":
    main()
