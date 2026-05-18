"""X-phase audit (X6) — compare prod-rasmachine-x against the categorical
baseline (prod-rasmachine-2026-05-14d) on the same 120-statement subset.

Emits each of G1–G5 with a numeric value + pass/fail flag.
"""
from __future__ import annotations

import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import duckdb

ROOT = Path(__file__).parent.parent
DB = ROOT / "data" / "corpus.duckdb"
NEW_RUN = "d026e3ec941e4adfab6aff87d02812ca"   # prod-rasmachine-x
OLD_RUN = "b05ddc85e43147d09ebc99ff7a164cf0"   # prod-rasmachine-2026-05-14d (baseline)

STRATA_FILE = ROOT / "data" / "corpora" / "rasmachine_subset_strata.jsonl"

PROBE_RE = re.compile(r"(subject_role|object_role|relation_axis|scope)=(\S+) \((\w+)\)")

random.seed(20260518)


def load_strata() -> dict[str, str]:
    out: dict[str, str] = {}
    with open(STRATA_FILE) as f:
        for line in f:
            r = json.loads(line)
            out[r["stmt_hash"]] = r["stratum"]
    return out


def fetch_aggregate(con, run_id: str, stmt_hashes: list[str]) -> dict[tuple[str, str], dict]:
    """Return {(stmt_hash, evidence_hash): aggregate_dict} for the run."""
    placeholders = ",".join("?" for _ in stmt_hashes)
    q = (
        f"SELECT stmt_hash, evidence_hash, output_json FROM scorer_step "
        f"WHERE run_id = ? AND step_kind = 'aggregate' "
        f"AND stmt_hash IN ({placeholders})"
    )
    rows = con.execute(q, [run_id] + stmt_hashes).fetchall()
    return {(sh, eh): json.loads(o) for sh, eh, o in rows}


def main():
    strata = load_strata()
    print(f"strata: {Counter(strata.values())}")
    print()

    con = duckdb.connect(str(DB), read_only=True)
    try:
        # ===== Pull aggregate rows for both runs on the subset stmts =====
        subset_hashes = list(strata.keys())
        new_agg = fetch_aggregate(con, NEW_RUN, subset_hashes)
        old_agg = fetch_aggregate(con, OLD_RUN, subset_hashes)
        print(f"new run rows: {len(new_agg)}")
        print(f"old run rows: {len(old_agg)}")
        # Intersection: same (stmt, evidence) pairs scored in BOTH runs
        shared = sorted(set(new_agg) & set(old_agg))
        print(f"shared (stmt, ev) pairs: {len(shared)}")
        print()

        # ===== G2: abstain count in new run =====
        n_abstain_new = sum(
            1 for d in new_agg.values() if d.get("verdict") == "abstain"
        )
        g2_pass = n_abstain_new == 0
        print(f"G2  abstain rows in new run:        {n_abstain_new:5d}   "
              f"[{'PASS' if g2_pass else 'FAIL'}]")

        # ===== Confusion matrix (old × new) on shared pairs =====
        confusion = Counter()
        for k in shared:
            ov, nv = old_agg[k]["verdict"], new_agg[k]["verdict"]
            confusion[(ov, nv)] += 1
        print()
        print("== confusion (old × new) ==")
        for ov in ("correct", "incorrect", "abstain"):
            for nv in ("correct", "incorrect"):
                n = confusion.get((ov, nv), 0)
                print(f"  old={ov:10s} → new={nv:10s}  {n:5d}")

        # ===== G1: alias rescue rate on absent_absent stratum =====
        # Rescued = was abstain w/ grounding_gap in old, is correct in new.
        absent_absent_hashes = [
            sh for sh, stratum in strata.items() if stratum == "absent_absent"
        ]
        aa_evidences = [k for k in shared if k[0] in set(absent_absent_hashes)]
        rescued = 0
        downgraded = 0
        stayed_low = 0
        for k in aa_evidences:
            old_v = old_agg[k]["verdict"]
            new_v = new_agg[k]["verdict"]
            new_s = new_agg[k].get("score", 0.0)
            if old_v == "abstain" and new_v == "correct":
                rescued += 1
            elif old_v == "abstain" and new_v == "incorrect":
                downgraded += 1
            elif old_v == "abstain":
                stayed_low += 1
        n_aa_old_abstain = sum(
            1 for k in aa_evidences if old_agg[k]["verdict"] == "abstain"
        )
        g1_rate = rescued / n_aa_old_abstain if n_aa_old_abstain else 0.0
        g1_pass = g1_rate >= 0.40
        print()
        print(f"G1  alias-fix rescue on absent_absent:")
        print(f"      old=abstain count:           {n_aa_old_abstain}")
        print(f"      rescued (→ correct):         {rescued}")
        print(f"      committed incorrect:         {downgraded}")
        print(f"      rescue rate:                 {g1_rate:.2%}   "
              f"[{'PASS' if g1_pass else 'FAIL'} ≥ 40%]")

        # ===== G3: regression rate on correct_high stratum =====
        ch_hashes = {sh for sh, st in strata.items() if st == "correct_high"}
        ch_pairs = [k for k in shared if k[0] in ch_hashes]
        flips = sum(
            1 for k in ch_pairs
            if old_agg[k]["verdict"] == "correct"
            and new_agg[k]["verdict"] == "incorrect"
        )
        n_ch_old_correct = sum(
            1 for k in ch_pairs if old_agg[k]["verdict"] == "correct"
        )
        g3_rate = flips / n_ch_old_correct if n_ch_old_correct else 0.0
        g3_pass = g3_rate <= 0.10
        print()
        print(f"G3  regression on correct_high:")
        print(f"      old=correct count:           {n_ch_old_correct}")
        print(f"      flipped to incorrect:        {flips}")
        print(f"      flip rate:                   {g3_rate:.2%}   "
              f"[{'PASS' if g3_pass else 'FAIL'} ≤ 10%]")

        # ===== G4: MAE vs INDRA published belief =====
        indra_priors = dict(con.execute("""
            SELECT target_id, CAST(value_text AS DOUBLE) FROM truth_label
            WHERE truth_set_id = 'indra_published_belief' AND target_kind = 'stmt'
        """).fetchall())
        new_deltas = []
        old_deltas = []
        for k, d in new_agg.items():
            sh = k[0]
            if sh not in indra_priors:
                continue
            s = d.get("score")
            if s is None:
                continue
            new_deltas.append(s - indra_priors[sh])
        for k, d in old_agg.items():
            sh = k[0]
            if sh not in indra_priors:
                continue
            s = d.get("score")
            if s is None:
                continue
            old_deltas.append(s - indra_priors[sh])
        if new_deltas:
            new_mae = statistics.mean(abs(x) for x in new_deltas)
            new_bias = statistics.mean(new_deltas)
            new_rmse = (sum(x*x for x in new_deltas) / len(new_deltas)) ** 0.5
        else:
            new_mae = new_bias = new_rmse = 0.0
        old_mae = statistics.mean(abs(x) for x in old_deltas) if old_deltas else 0.0
        g4_pass = new_mae <= 0.45  # baseline 0.398; allow 12% slack
        print()
        print(f"G4  calibration vs INDRA prior (proxy, not gold):")
        print(f"      MAE (new):                   {new_mae:.3f}   (old: {old_mae:.3f})")
        print(f"      bias (new):                  {new_bias:+.3f}")
        print(f"      RMSE (new):                  {new_rmse:.3f}")
        print(f"      pass: new_MAE ≤ 0.45         "
              f"[{'PASS' if g4_pass else 'FAIL'}]")

        # ===== Verdict-share distribution =====
        new_verdicts = Counter(d["verdict"] for d in new_agg.values())
        new_score_buckets = Counter()
        for d in new_agg.values():
            s = d.get("score") or 0.0
            if s >= 0.95:   b = "0.95-1.00"
            elif s >= 0.80: b = "0.80-0.95"
            elif s >= 0.65: b = "0.65-0.80"
            elif s >= 0.50: b = "0.50-0.65"
            elif s >= 0.35: b = "0.35-0.50"
            elif s >= 0.20: b = "0.20-0.35"
            elif s >= 0.05: b = "0.05-0.20"
            else:           b = "<0.05"
            new_score_buckets[b] += 1
        print()
        print("== new verdict shares ==")
        for v, n in new_verdicts.most_common():
            print(f"  {v:10s} {n:5d}  ({100*n/sum(new_verdicts.values()):5.1f}%)")
        print()
        print("== new score distribution ==")
        for b in ["<0.05","0.05-0.20","0.20-0.35","0.35-0.50",
                  "0.50-0.65","0.65-0.80","0.80-0.95","0.95-1.00"]:
            n = new_score_buckets[b]
            bar = "█" * int(50 * n / sum(new_score_buckets.values()))
            print(f"  {b:11s} {n:5d}  {bar}")

        # ===== G5: hand-audit pool =====
        rescued_pool = [
            k for k in aa_evidences
            if old_agg[k]["verdict"] == "abstain"
            and new_agg[k]["verdict"] == "correct"
        ]
        regression_pool = [
            k for k in ch_pairs
            if old_agg[k]["verdict"] == "correct"
            and new_agg[k]["verdict"] == "incorrect"
        ]
        n_audit_rescued = min(15, len(rescued_pool))
        n_audit_regression = min(10, len(regression_pool))
        audit_sample = random.sample(rescued_pool, n_audit_rescued) + \
                       random.sample(regression_pool, n_audit_regression)

        print()
        print(f"G5  hand-audit pool:")
        print(f"      rescued-pool size:           {len(rescued_pool)}")
        print(f"      regression-pool size:        {len(regression_pool)}")
        print(f"      sample for audit:            "
              f"{n_audit_rescued} rescued + {n_audit_regression} regressions")

        # Write audit-target file for manual inspection
        audit_path = ROOT / "data" / "results" / "x_phase_audit_pool.jsonl"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(audit_path, "w") as f:
            for k in audit_sample:
                sh, eh = k
                old_d = old_agg[k]
                new_d = new_agg[k]
                # Pull evidence text + agents
                ev_row = con.execute("""
                    SELECT e.text, e.source_api, s.indra_type, s.raw_json
                    FROM evidence e JOIN statement s ON s.stmt_hash = e.stmt_hash
                    WHERE e.evidence_hash = ? AND s.stmt_hash = ?
                """, [eh, sh]).fetchone()
                f.write(json.dumps({
                    "audit_category": "rescued" if k in rescued_pool else "regression",
                    "stmt_hash": sh,
                    "evidence_hash": eh,
                    "stratum": strata.get(sh),
                    "old_verdict": old_d.get("verdict"),
                    "old_score": old_d.get("score"),
                    "old_reasons": old_d.get("reasons"),
                    "new_verdict": new_d.get("verdict"),
                    "new_score": new_d.get("score"),
                    "new_reasons": new_d.get("reasons"),
                    "indra_type": ev_row[2] if ev_row else None,
                    "evidence_text": ev_row[0] if ev_row else None,
                    "source_api": ev_row[1] if ev_row else None,
                    "raw_stmt": ev_row[3] if ev_row else None,
                }) + "\n")
        print(f"      wrote audit pool to:         {audit_path}")
    finally:
        con.close()

    print()
    print("=" * 60)
    print("=== X6 SUMMARY ===")
    print(f"  G1 alias rescue rate (≥40%):    {'PASS' if g1_pass else 'FAIL'}  ({g1_rate:.2%})")
    print(f"  G2 zero abstain:                {'PASS' if g2_pass else 'FAIL'}  ({n_abstain_new})")
    print(f"  G3 regression rate (≤10%):      {'PASS' if g3_pass else 'FAIL'}  ({g3_rate:.2%})")
    print(f"  G4 MAE vs INDRA (≤0.45):        {'PASS' if g4_pass else 'FAIL'}  ({new_mae:.3f})")
    print(f"  G5 hand audit:                  MANUAL — review audit_pool.jsonl")
    print("=" * 60)


if __name__ == "__main__":
    main()
