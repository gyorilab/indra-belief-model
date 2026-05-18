"""X-phase: stratified subset selection from existing rasmachine run.

Reads run b05ddc85e43147d09ebc99ff7a164cf0 from corpus.duckdb, classifies each
(stmt, evidence) pair by stratum (alias-fix target / regression guard / etc.),
selects 120 statements, and emits:
  - data/corpora/rasmachine_subset.json — INDRA Statement JSON for the worker
  - data/corpora/rasmachine_subset_strata.jsonl — per-stmt stratum tag for audit
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import duckdb

ROOT = Path(__file__).parent.parent
DB = ROOT / "data" / "corpus.duckdb"
RUN_ID = "b05ddc85e43147d09ebc99ff7a164cf0"
OUT_JSON = ROOT / "data" / "corpora" / "rasmachine_subset.json"
OUT_STRATA = ROOT / "data" / "corpora" / "rasmachine_subset_strata.jsonl"

# Subset shape — total 120 stmts
STRATA_TARGETS = {
    "absent_absent":      40,   # Tier 1 alias-fix primary target
    "ra_abstain":         15,   # log-odds combiner test
    "indirect_chain":     10,   # mediator factor test
    "correct_high":       25,   # regression guard
    "incorrect_clean":    20,   # regression guard
    "scope_hedged_neg":   10,   # scope multiplier validation
}

PROBE_RE = re.compile(r"(subject_role|object_role|relation_axis|scope)=(\S+) \(\w+\)")

random.seed(20260518)


def classify(aggregate_out: dict) -> str | None:
    """Bucket an aggregate-row's joint probe state into a stratum."""
    rt = aggregate_out.get("raw_text", "")
    probes = {m[0]: m[1] for m in PROBE_RE.findall(rt)}
    sr = probes.get("subject_role"); or_ = probes.get("object_role")
    ra = probes.get("relation_axis"); sc = probes.get("scope")
    v = aggregate_out.get("verdict")
    s = aggregate_out.get("score") or 0.0
    reasons = aggregate_out.get("reasons") or []

    # Mediator stratum gets priority — small but distinct.
    if (sr in ("present_as_mediator", "via_mediator")
            or or_ in ("present_as_mediator", "via_mediator")):
        return "indirect_chain"
    if sc in ("hedged", "negated"):
        return "scope_hedged_neg"
    # absent-absent grounding gap — the headline cluster
    if sr == "absent" and or_ == "absent":
        return "absent_absent"
    # ra=abstain when entities are present — log-odds combiner stress test
    if ra == "abstain" and sr != "absent" and or_ != "absent":
        return "ra_abstain"
    if v == "correct" and s >= 0.80:
        return "correct_high"
    if v == "incorrect" and s <= 0.20 and "absent_relationship" in reasons:
        return "incorrect_clean"
    return None


def main():
    con = duckdb.connect(str(DB), read_only=True)
    try:
        rows = con.execute(
            "SELECT stmt_hash, output_json FROM scorer_step "
            "WHERE run_id = ? AND step_kind = 'aggregate'",
            [RUN_ID],
        ).fetchall()
        print(f"loaded {len(rows)} aggregate rows from run {RUN_ID[:8]}")

        # Group: stmt_hash → list of strata its evidences land in
        stmt_strata: dict[str, Counter] = defaultdict(Counter)
        for stmt_hash, out_json in rows:
            d = json.loads(out_json)
            bucket = classify(d)
            if bucket is None:
                continue
            stmt_strata[stmt_hash][bucket] += 1

        print(f"stmts with at least one classifiable evidence: {len(stmt_strata)}")
        per_bucket_stmts: dict[str, list[str]] = defaultdict(list)
        for sh, ctr in stmt_strata.items():
            dominant = ctr.most_common(1)[0][0]
            per_bucket_stmts[dominant].append(sh)
        print()
        print("dominant-stratum membership:")
        for b, members in sorted(per_bucket_stmts.items(), key=lambda x: -len(x[1])):
            print(f"  {b:20s} {len(members):5d}  (need {STRATA_TARGETS.get(b, 0)})")

        # Sample per target
        picked: list[tuple[str, str]] = []
        for stratum, n_want in STRATA_TARGETS.items():
            pool = per_bucket_stmts.get(stratum, [])
            if not pool:
                print(f"WARNING: empty pool for {stratum}")
                continue
            n = min(n_want, len(pool))
            for sh in random.sample(pool, n):
                picked.append((sh, stratum))
        print()
        print(f"selected {len(picked)} statements")

        # Pull raw_json + write subset JSON
        sh_list = [sh for sh, _ in picked]
        placeholders = ",".join("?" for _ in sh_list)
        raw_rows = con.execute(
            f"SELECT stmt_hash, raw_json FROM statement WHERE stmt_hash IN ({placeholders})",
            sh_list,
        ).fetchall()
        raw_by_hash = {r[0]: json.loads(r[1]) for r in raw_rows}

        # Preserve picked-order with strata
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        stmts_out = [raw_by_hash[sh] for sh, _ in picked if sh in raw_by_hash]
        with open(OUT_JSON, "w") as f:
            json.dump(stmts_out, f)
        print(f"wrote {OUT_JSON} ({len(stmts_out)} statements)")

        with open(OUT_STRATA, "w") as f:
            for sh, stratum in picked:
                if sh in raw_by_hash:
                    f.write(json.dumps({"stmt_hash": sh, "stratum": stratum}) + "\n")
        print(f"wrote {OUT_STRATA}")

        # Total evidence count (sanity)
        ev_count = con.execute(
            f"SELECT COUNT(*) FROM evidence WHERE stmt_hash IN ({placeholders})",
            sh_list,
        ).fetchone()[0]
        print(f"total evidences in subset: {ev_count} "
              f"(expect ~{ev_count*6.3/60:.0f} min walltime on gemma-remote)")
    finally:
        con.close()


if __name__ == "__main__":
    main()
