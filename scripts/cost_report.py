"""What did the scoring cost, and what would the rest of the fleet cost?

Actual spend = sum over each run's call_log (prompt_tokens, out_tokens) x the
list/estimate price (corpus.cost.price_for, keyed on the call_log model_id).

Projection for a model we have NOT run on the external gold = that model's OWN
observed per-evidence (in, out) token profile from its rasmachine_v1 run, scaled
to the external gold's 587 evidences x its price. Output dominates cost and varies
wildly by model (reasoners emit long CoT), so using each model's own profile is the
honest estimate.

    PYTHONPATH=src .venv/bin/python scripts/cost_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.corpus.cost import price_for  # noqa: E402

RES = ROOT / "data" / "results"
EXT_EV = 587  # evidences in the external curator gold

# what I ran THIS session (bedrock)
SESSION = [
    "rasmachine_v2_bedrock-gemma", "rasmachine_v2_bedrock-nemotron-nano-30b",
    "external_curator_v1_bedrock-gemma", "external_curator_v1_bedrock-nemotron-nano-30b",
]


def tally(path: Path):
    """(model_id, in_tokens, out_tokens, n_evidence_rows) for a run file."""
    mid, ti, to, n = None, 0, 0, 0
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        n += 1
        for c in (r.get("call_log") or []):
            mid = c.get("model_id") or mid
            ti += c.get("prompt_tokens") or 0
            to += c.get("out_tokens") or 0
    return mid, ti, to, n


def cost(mid, ti, to):
    p = price_for(mid)
    if not p:
        return None, "NOPRICE"
    return ti * p[0] / 1e6 + to * p[1] / 1e6, p[2]


def main():
    print("=== ACTUAL SPEND THIS SESSION ===")
    print(f"{'run':>44} {'model_id':>30} {'in_tok':>9} {'out_tok':>9} {'USD':>8}")
    total = 0.0
    for name in SESSION:
        p = RES / f"{name}.jsonl"
        if not p.exists():
            print(f"{name:>44}  (missing)"); continue
        mid, ti, to, n = tally(p)
        c, basis = cost(mid, ti, to)
        total += c or 0
        print(f"{name:>44} {mid:>30} {ti:9,} {to:9,} {('$%.3f'%c) if c is not None else basis:>8}")
    print(f"{'TOTAL session bedrock spend':>96} ${total:.2f}")

    # ── projection: every other previously-run bedrock model on 587 ev ──
    print(f"\n=== PROJECTED COST to run the OTHER fleet on the external gold ({EXT_EV} ev) ===")
    print("  (each model's OWN per-evidence token profile from its rasmachine_v1 run x price)")
    print(f"{'model':>26} {'in/ev':>7} {'out/ev':>7} {'basis':>9} {'$/run(587ev)':>13}")
    seen, rows, proj_total = set(), [], 0.0
    for p in sorted(RES.glob("rasmachine_v1_bedrock-*.jsonl")):
        if "progress" in p.name:
            continue
        mid, ti, to, n = tally(p)
        if not n or mid is None or mid in seen:
            continue
        seen.add(mid)
        in_ev, out_ev = ti / n, to / n
        pr = price_for(mid)
        if not pr:
            rows.append((mid, in_ev, out_ev, "NOPRICE", None)); continue
        run_cost = (in_ev * pr[0] + out_ev * pr[1]) * EXT_EV / 1e6
        rows.append((mid, in_ev, out_ev, pr[2], run_cost))
    # already-run-on-external (skip from "other") -> note separately
    already = {"google.gemma-4-26b-a4b", "nvidia.nemotron-nano-3-30b"}
    for mid, ie, oe, basis, c in sorted(rows, key=lambda r: (r[4] is None, r[4] or 0)):
        flag = "  <- already ran on external" if mid in already else ""
        cstr = f"${c:.3f}" if c is not None else basis
        if mid not in already and c is not None:
            proj_total += c
        print(f"{mid:>26} {ie:7.0f} {oe:7.0f} {basis:>9} {cstr:>13}{flag}")
    print(f"\n  SUM to run ALL others (excl. gemma+nemotron already done): ${proj_total:.2f}")
    print(f"  (add the 2 already done: ${proj_total:.2f} covers the remaining fleet)")


if __name__ == "__main__":
    main()
