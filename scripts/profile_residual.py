"""Profile the error that REMAINS after the calibrated soft belief.

Once aggregation (gated noisy-OR) + calibration (verdict-conditioned w) are
in, the question is: what's left, and is it OURS to fix? This decomposes every
statement-level belief error into:

  - verdict-driven : at least one per-evidence VERDICT disagreed with the gold on
                     that evidence -> the error is upstream (the reader), NOT the
                     belief math. Reducible only by a better reader/grounding.
  - aggregation    : every per-evidence verdict MATCHED its gold, yet the
                     statement rolled up to the wrong side -> the error is in OUR
                     belief/threshold. Reducible by us.

and splits the verdict-driven errors by the bucket taxonomy (reader_hallucination
/ schema-artifact / genuine semantic) so we see how much is model noise vs real.
Also dumps the most-confident errors (|belief-0.5| largest) for human/agent
categorisation. In-sample on eval_curation_v1 (the held-out picture comes with
the holdout_cc run).

    PYTHONPATH=src python scripts/profile_residual.py --model gemma-remote
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

import calibration_stage0 as c0  # noqa: E402
from indra_belief.calibration_constants import calibration_for  # noqa: E402
from indra_belief.curation import aggregate_gold, is_gold_correct  # noqa: E402
from indra_belief.results import HALLUC, HEDGE, PLACEHOLDER_MAX_LEN  # noqa: E402
from indra_belief.statement_belief import statement_belief  # noqa: E402


def ev_bucket(verdict, tag, text) -> str:
    """Coarse bucket for ONE evidence whose verdict disagrees with its gold."""
    tl = len(text or "")
    if tl == 0:
        return "no_text"
    if tl < PLACEHOLDER_MAX_LEN:
        return "placeholder_text"
    if verdict == "incorrect" and is_gold_correct(tag):
        # model said wrong, gold says right -> a false alarm (over-flag)
        return "reader_hallucination_overflag"
    if verdict == "correct" and not is_gold_correct(tag):
        return "missed_error"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/benchmark/eval_curation_v1.jsonl")
    ap.add_argument("--run", default="data/results/eval_curation_v1_gemma.jsonl")
    ap.add_argument("--model", default="gemma-remote")
    ap.add_argument("--label", default=None)
    ap.add_argument("--dump", default="data/results/residual_examples.json")
    ap.add_argument("--out", default="data/results/residual_profile.md")
    ap.add_argument("--dump-n", type=int, default=60)
    args = ap.parse_args()
    label = args.label or args.model
    calib = calibration_for(args.model)

    gold = c0.load_jsonl(ROOT / args.gold)
    by_pair, by_sh = c0.build_gold_index(gold)
    rows = c0.load_jsonl(ROOT / args.run)
    joined, _, _ = c0.join_model(rows, by_pair, by_sh)

    # group by statement, carrying per-evidence detail
    by_stmt: dict = defaultdict(lambda: {"ev": [], "tags": [], "claim": None})
    for g, s in joined:
        rec = by_stmt[g["pa_hash"]]
        rec["ev"].append({
            "source_api": s.get("source_api") or g.get("source_api"),
            "verdict": s.get("verdict"), "confidence": s.get("confidence"),
            "tier": s.get("tier"), "tag": g["tag"], "text": g.get("evidence_text") or "",
        })
        rec["tags"].append(g["tag"])
        if rec["claim"] is None:
            rec["claim"] = {"subject": g.get("subject"), "stmt_type": g.get("stmt_type"),
                            "object": g.get("object")}

    stmts = []
    for pa, rec in by_stmt.items():
        gold_agg = aggregate_gold(rec["tags"])
        if gold_agg is None:
            continue
        gc = is_gold_correct(gold_agg)
        sb = statement_belief(rec["ev"], soft=calib)
        belief = sb.belief
        if belief is None:
            continue
        sys_correct = belief >= 0.5
        n_verdict_wrong = sum(1 for e in rec["ev"]
                              if (e["verdict"] == "correct") != is_gold_correct(e["tag"]))
        stmts.append({"pa_hash": pa, "claim": rec["claim"], "ev": rec["ev"],
                      "gold_correct": gc, "belief": belief, "sys_correct": sys_correct,
                      "n_ev": len(rec["ev"]), "n_verdict_wrong": n_verdict_wrong,
                      "verdict_statement": sb.verdict_statement})

    errors = [s for s in stmts if s["sys_correct"] != s["gold_correct"]]
    fp = [s for s in errors if not s["gold_correct"] and s["sys_correct"]]   # believed a wrong stmt
    fn = [s for s in errors if s["gold_correct"] and not s["sys_correct"]]   # doubted a right stmt

    # decomposition: verdict-driven vs aggregation
    verdict_driven = [s for s in errors if s["n_verdict_wrong"] > 0]
    aggregation = [s for s in errors if s["n_verdict_wrong"] == 0]

    # bucket the verdict-driven errors by their wrong evidences
    bkt = Counter()
    for s in verdict_driven:
        for e in s["ev"]:
            if (e["verdict"] == "correct") != is_gold_correct(e["tag"]):
                bkt[ev_bucket(e["verdict"], e["tag"], e["text"])] += 1

    by_type = Counter(s["claim"]["stmt_type"] for s in errors)
    by_depth = Counter("single" if s["n_ev"] == 1 else "multi" for s in errors)

    def pct(n, d):
        return f"{100*n/d:.1f}%" if d else "—"

    L = [f"# Residual profile — calibrated soft belief ({label}, in-sample eval_curation_v1)\n",
         f"Statements {len(stmts)} · belief errors **{len(errors)}** "
         f"({pct(len(errors), len(stmts))}): false-confidence (believed a wrong stmt) {len(fp)}, "
         f"false-doubt (doubted a right stmt) {len(fn)}.\n",
         "## Where the error lives\n",
         "| class | n | share of errors | meaning |",
         "|---|--:|--:|---|",
         f"| verdict-driven | {len(verdict_driven)} | {pct(len(verdict_driven), len(errors))} | "
         "a per-evidence VERDICT was wrong → upstream (reader), not our belief math |",
         f"| aggregation | {len(aggregation)} | {pct(len(aggregation), len(errors))} | "
         "every verdict matched gold, yet the statement rolled up wrong → OURS to fix |",
         "",
         "## Verdict-driven errors by kind (the wrong evidences)\n",
         "| kind | n |", "|---|--:|"]
    for k, n in bkt.most_common():
        L.append(f"| {k} | {n} |")
    L += ["", "## Errors by statement type\n", "| stmt_type | n |", "|---|--:|"]
    for k, n in by_type.most_common():
        L.append(f"| {k} | {n} |")
    L += ["", f"By depth: single-evidence {by_depth['single']}, multi-evidence {by_depth['multi']}.",
          "", "> verdict-driven = the calibration/aggregation did its job; the residual is upstream "
          "reader error (per D8, the lever is reader/grounding quality). aggregation = the part our "
          "belief math could still recover.\n"]
    (ROOT / args.out).write_text("\n".join(L))

    # dump most-confident errors for categorisation (claim + evidence + gold + verdict)
    errors.sort(key=lambda s: abs(s["belief"] - 0.5), reverse=True)
    dump = []
    for s in errors[:args.dump_n]:
        c = s["claim"]
        dump.append({
            "pa_hash": s["pa_hash"],
            "claim": f"{c['subject']} [{c['stmt_type']}] {c['object']}",
            "gold_correct": s["gold_correct"], "belief": round(s["belief"], 3),
            "error": "false_confidence" if s["sys_correct"] else "false_doubt",
            "n_ev": s["n_ev"], "n_verdict_wrong": s["n_verdict_wrong"],
            "where": "verdict-driven" if s["n_verdict_wrong"] else "aggregation",
            "evidence": [{"source": e["source_api"], "verdict": e["verdict"],
                          "confidence": e["confidence"], "gold_tag": e["tag"],
                          "text": e["text"]} for e in s["ev"]],
        })
    (ROOT / args.dump).write_text(json.dumps(dump, indent=2))

    print("\n".join(L))
    print(f"\nwrote {args.out} and {args.dump} ({len(dump)} examples)")


if __name__ == "__main__":
    main()
