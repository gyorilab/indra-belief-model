"""Calibration Stage C0 — honest reliability curve, zero model change.

The decision gate (G0) for the whole calibration arc. See
``research/calibration_task_hypergraph.md``. Touches NO production code path —
it joins the existing eval_curation_v1 run outputs to gold and asks three
questions:

  C0.1  Verdict-conditional error anchors per reader (rand_corr / rand_rej) —
        the two parameters Stage C1 would fit. Confidence axis is reported but
        NOT fit (it has collapsed; see the printed mix).
  C0.2  Tier-1 per-EVIDENCE reliability of the persisted grid ``score`` as a
        forecast of ``gold == correct`` — reliability bins + ECE + Brier
        (Murphy decomposition). Expected to be degenerate (~2 occupied bins)
        because the grid score is near-binary once confidence collapses.
  C0.3  Tier-2 per-STATEMENT raw-belief headroom — the DECISIVE test. Aggregate
        the per-evidence verdicts through the production noise model
        (``compute_gated_belief`` + RECALIBRATED_PRIORS, hard gate
        ``included = verdict != 'incorrect'``) to a per-statement belief, then
        measure AUROC / AUPRC vs per-statement gold (any-incorrect-wins).

G0 read: if the per-statement belief AUPRC has no headroom over the base rate
(and AUROC ~ 0.5), a monotone post-hoc map cannot manufacture discrimination —
the lever is upstream (reader / grounding), and Stages C1-C3 should not fire.

Join is the canonical (matches_hash, source_hash) PAIR used by
eval_curation_compare.py (source_hash alone is not unique — version-skew lesson).

    PYTHONPATH=src python scripts/calibration_stage0.py \
        --gold data/benchmark/eval_curation_v1.jsonl \
        --run data/results/eval_curation_v1_medpsy.jsonl --name MedPsy-4B \
        --run data/results/eval_curation_v1_gemma.jsonl  --name gemma-26B \
        --out data/results/calibration_stage0.md \
        --json data/results/calibration_stage0.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.curation import aggregate_gold, is_gold_correct  # noqa: E402
from indra_belief.metrics import BINS_8, ece  # noqa: E402
from indra_belief.noise_model import (  # noqa: E402
    RECALIBRATED_PRIORS,
    compute_gated_belief,
)

MASK = (1 << 64) - 1


def umask(x) -> int:
    return int(x) & MASK


def load_jsonl(p: str | Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


# ---- canonical join (mirrors eval_curation_compare.py) ----------------------
def build_gold_index(gold_rows: list[dict]):
    by_pair: dict[tuple[int, int], dict] = {}
    by_sh: dict[int, list[dict]] = defaultdict(list)
    for r in gold_rows:
        mh = umask(r["matches_hash"])
        sh = umask(r["source_hash"])
        by_pair[(mh, sh)] = r
        by_sh[sh].append(r)
    return by_pair, by_sh


def gold_for(scored: dict, by_pair, by_sh) -> dict | None:
    sh = umask(scored["source_hash"])
    stmt_hash_hex = scored.get("stmt_hash")
    mh = int(stmt_hash_hex, 16) if stmt_hash_hex else None
    if mh is not None and (mh, sh) in by_pair:
        return by_pair[(mh, sh)]
    cand = by_sh.get(sh, [])
    return cand[0] if len(cand) == 1 else None


def join_model(scored_rows, by_pair, by_sh):
    joined, parse_null, missed = [], 0, 0
    for s in scored_rows:
        g = gold_for(s, by_pair, by_sh)
        if g is None:
            missed += 1
            continue
        if s.get("verdict") is None:
            parse_null += 1
            continue
        joined.append((g, s))
    return joined, parse_null, missed


# ---- metrics not in the shared lib (AUROC / AUPRC / Brier) -------------------
def _rankdata_avg(a: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged (no scipy)."""
    order = a.argsort(kind="mergesort")
    sa = a[order]
    n = len(a)
    rnk = np.empty(n, float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        rnk[i : j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    out = np.empty(n, float)
    out[order] = rnk
    return out


def auroc(scores, labels) -> float:
    """AUROC via Mann-Whitney U. positive class = label True."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata_avg(s)
    return (ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auprc(scores, labels) -> float:
    """Average precision (AUPRC), positive class = label True."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    # AP = sum over thresholds of (R_k - R_{k-1}) * P_k
    rec_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - rec_prev) * precision))


def brier_murphy(scores, labels, bins=BINS_8) -> dict:
    """Brier score + Murphy decomposition (reliability - resolution + uncertainty)."""
    p = np.asarray(scores, float)
    y = np.asarray(labels, float)
    n = len(p)
    if n == 0:
        return {"brier": float("nan"), "reliability": float("nan"),
                "resolution": float("nan"), "uncertainty": float("nan"), "n": 0}
    brier = float(np.mean((p - y) ** 2))
    ybar = float(np.mean(y))
    uncertainty = ybar * (1.0 - ybar)
    reliability = 0.0
    resolution = 0.0
    for lo, hi in bins:
        m = (p >= lo) & (p < hi)
        nk = int(m.sum())
        if nk == 0:
            continue
        pbar_k = float(np.mean(p[m]))
        ybar_k = float(np.mean(y[m]))
        reliability += nk / n * (pbar_k - ybar_k) ** 2
        resolution += nk / n * (ybar_k - ybar) ** 2
    return {"brier": brier, "reliability": reliability,
            "resolution": resolution, "uncertainty": uncertainty, "n": n}


def reliability_bins(scores, labels, bins=BINS_8) -> list[dict]:
    p = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    out = []
    for lo, hi in bins:
        m = (p >= lo) & (p < hi)
        nk = int(m.sum())
        out.append({
            "lo": lo, "hi": hi, "n": nk,
            "mean_pred": float(np.mean(p[m])) if nk else None,
            "empirical": float(np.mean(y[m])) if nk else None,
        })
    return out


# ---- per-statement aggregation (Tier-2) -------------------------------------
def per_statement_belief(joined) -> list[dict]:
    """Aggregate joined (gold, scored) rows to one row per statement.

    belief = compute_gated_belief over the statement's scored evidences with
    the production hard gate (included = verdict != 'incorrect') and
    RECALIBRATED_PRIORS. gold = any-incorrect-wins over the statement's tags.
    """
    # Group into statements by pa_hash (the preassembled-statement unit the rest
    # of the calibration arc uses, C2.4) — not matches_hash, which is finer.
    by_stmt: dict[int, dict] = defaultdict(lambda: {"ev": [], "tags": [], "indra_belief": None})
    for g, s in joined:
        mh = g["pa_hash"]
        rec = by_stmt[mh]
        rec["ev"].append({
            "source_api": s.get("source_api") or g.get("source_api"),
            "included": s.get("verdict") != "incorrect",
        })
        rec["tags"].append(g["tag"])
        if rec["indra_belief"] is None:
            rec["indra_belief"] = s.get("belief")
    rows = []
    for mh, rec in by_stmt.items():
        gold = aggregate_gold(rec["tags"])
        if gold is None:
            continue
        res = compute_gated_belief(rec["ev"], priors=RECALIBRATED_PRIORS)
        rows.append({
            "pa_hash": mh,
            "belief": res.belief,
            "parametric_only": res.parametric_only,
            "indra_belief": rec["indra_belief"],
            "n_ev": len(rec["ev"]),
            "gold_correct": is_gold_correct(gold),
        })
    return rows


# ---- per-model analysis ------------------------------------------------------
def analyze(name: str, joined, parse_null, missed) -> dict:
    # C0.1 — verdict-conditional anchors
    cells = defaultdict(int)  # (verdict, gold_correct) -> n
    conf = defaultdict(int)
    for g, s in joined:
        gc = is_gold_correct(g["tag"])
        cells[(s["verdict"], gc)] += 1
        conf[s.get("confidence")] += 1
    cc = cells[("correct", True)]
    ci = cells[("correct", False)]
    ic = cells[("incorrect", True)]
    ii = cells[("incorrect", False)]
    rand_corr = ci / (ci + cc) if (ci + cc) else float("nan")   # P(gold=incorrect|verdict=correct)
    rand_rej = ic / (ic + ii) if (ic + ii) else float("nan")    # P(gold=correct|verdict=incorrect)

    # C0.2 — Tier-1 per-evidence reliability of the grid score
    ev_pairs = [(s.get("score") if s.get("score") is not None else 0.5,
                 is_gold_correct(g["tag"])) for g, s in joined]
    ev_scores = [p for p, _ in ev_pairs]
    ev_labels = [y for _, y in ev_pairs]
    t1 = {
        "ece": ece(ev_pairs),
        "auroc": auroc(ev_scores, ev_labels),
        "auprc_correct": auprc(ev_scores, ev_labels),
        "base_rate_correct": float(np.mean(ev_labels)),
        "brier": brier_murphy(ev_scores, ev_labels),
        "bins": reliability_bins(ev_scores, ev_labels),
        "n": len(ev_pairs),
    }

    # C0.3 — Tier-2 per-statement raw-belief headroom (DECISIVE)
    stmt = per_statement_belief(joined)
    bel = [r["belief"] for r in stmt]
    lab = [r["gold_correct"] for r in stmt]
    indra = [r["indra_belief"] for r in stmt if r["indra_belief"] is not None]
    indra_lab = [r["gold_correct"] for r in stmt if r["indra_belief"] is not None]
    singles = [r for r in stmt if r["n_ev"] == 1]
    multi = [r for r in stmt if r["n_ev"] > 1]

    def headroom(scores, labels):
        if not scores:
            return None
        labels_b = np.asarray(labels, bool)
        return {
            "n": len(scores),
            "base_rate_correct": float(labels_b.mean()),
            "auroc": auroc(scores, labels),
            "auprc_correct": auprc(scores, labels),
            "ece": ece(list(zip(scores, [bool(x) for x in labels]))),
            "brier": brier_murphy(scores, labels),
            "mean_belief_when_correct": float(np.mean([s for s, y in zip(scores, labels) if y])) if any(labels) else None,
            "mean_belief_when_incorrect": float(np.mean([s for s, y in zip(scores, labels) if not y])) if not all(labels) else None,
        }

    t2 = {
        "all": headroom(bel, lab),
        "bins": reliability_bins(bel, lab),
        "singletons": headroom([r["belief"] for r in singles], [r["gold_correct"] for r in singles]),
        "multi_evidence": headroom([r["belief"] for r in multi], [r["gold_correct"] for r in multi]),
        "indra_prior_reference": headroom(indra, indra_lab),
        "n_statements": len(stmt),
        "n_singletons": len(singles),
        "n_multi": len(multi),
    }

    # G0 verdict per model: headroom = AUROC meaningfully > 0.5 AND AUPRC > base rate
    a = t2["all"]
    headroom_auroc = (a["auroc"] - 0.5) if a else float("nan")
    headroom_auprc = (a["auprc_correct"] - a["base_rate_correct"]) if a else float("nan")
    go = bool(a and headroom_auroc > 0.03 and headroom_auprc > 0.02)

    return {
        "name": name,
        "joined": len(joined), "parse_null": parse_null, "missed": missed,
        "confidence_mix": dict(conf),
        "anchors": {
            "cells": {"correct_correct": cc, "correct_incorrect": ci,
                      "incorrect_correct": ic, "incorrect_incorrect": ii},
            "rand_corr": rand_corr, "rand_rej": rand_rej,
        },
        "tier1_per_evidence": t1,
        "tier2_per_statement": t2,
        "g0": {"headroom_auroc": headroom_auroc, "headroom_auprc": headroom_auprc, "go": go},
    }


# ---- rendering ---------------------------------------------------------------
def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def reliability_ascii(bins, width=24) -> list[str]:
    """One row per occupied bin: range, n, predicted vs empirical on a 0..1 axis."""
    lines = ["    bin            n     pred   emp    0" + " " * (width - 2) + "1"]
    for b in bins:
        if not b["n"]:
            continue
        pred, emp = b["mean_pred"], b["empirical"]
        row = [" "] * width
        pp = min(width - 1, int(round(pred * (width - 1))))
        ee = min(width - 1, int(round(emp * (width - 1))))
        row[pp] = "P"
        row[ee] = "E" if ee != pp else "X"
        lines.append(f"    [{b['lo']:.2f},{b['hi']:.2f}) {b['n']:>5}  {pred:.3f}  {emp:.3f}  |{''.join(row)}|")
    return lines


def render_md(results: list[dict], args) -> str:
    L = []
    e = L.append
    e("# Calibration Stage C0 — reliability curve (zero model change)")
    e("")
    e("Diagnostic for **G0** (go/no-go). No production code path touched. "
      "Per-evidence join on the canonical (matches_hash, source_hash) pair; "
      "per-statement belief = `compute_gated_belief` over scored evidences with "
      "the production hard gate (`included = verdict != incorrect`) + "
      "RECALIBRATED_PRIORS; per-statement gold = any-incorrect-wins. "
      "Generated by `scripts/calibration_stage0.py`.")
    e("")
    e("## G0 — go/no-go (does per-statement raw belief have headroom?)")
    e("")
    e("| reader | stmts | base rate | belief AUROC | Δ vs 0.5 | belief AUPRC | Δ vs base | verdict |")
    e("|---|---|---|---|---|---|---|---|")
    for r in results:
        a = r["tier2_per_statement"]["all"]
        g = r["g0"]
        verdict = "**GO**" if g["go"] else "**NO-GO**"
        e(f"| {r['name']} | {a['n']} | {_fmt(a['base_rate_correct'])} | {_fmt(a['auroc'])} | "
          f"{_fmt(g['headroom_auroc'],3)} | {_fmt(a['auprc_correct'])} | {_fmt(g['headroom_auprc'],3)} | {verdict} |")
    e("")
    e("> GO criterion: AUROC − 0.5 > 0.03 AND AUPRC − base_rate > 0.02. A NO-GO means a "
      "monotone post-hoc map cannot manufacture discrimination — the lever is upstream "
      "(reader / grounding), and Stages C1–C3 should not fire (D8).")
    e("")

    for r in results:
        e(f"## {r['name']}")
        e("")
        e(f"Joined {r['joined']} evidence rows (parse-null {r['parse_null']}, unmatched {r['missed']}). "
          f"Confidence mix: {r['confidence_mix']} — **degenerate; not fit (C0.1)**.")
        e("")
        an = r["anchors"]; c = an["cells"]
        e("### C0.1 — verdict-conditional anchors (the two params C1 would fit)")
        e("")
        e(f"- confusion (verdict × gold): correct/correct {c['correct_correct']}, "
          f"correct/incorrect {c['correct_incorrect']}, incorrect/correct {c['incorrect_correct']}, "
          f"incorrect/incorrect {c['incorrect_incorrect']}")
        e(f"- **rand_corr** = P(gold=incorrect | verdict=correct) = **{_fmt(an['rand_corr'])}** "
          f"(a confirmed read is wrong this often)")
        e(f"- **rand_rej** = P(gold=correct | verdict=incorrect) = **{_fmt(an['rand_rej'])}** "
          f"(a rejected read is wrong this often)")
        e("")
        t1 = r["tier1_per_evidence"]
        e("### C0.2 — Tier-1 per-evidence reliability of the grid `score`")
        e("")
        e(f"- ECE {_fmt(t1['ece'])} · Brier {_fmt(t1['brier']['brier'])} "
          f"(reliability {_fmt(t1['brier']['reliability'])} − resolution {_fmt(t1['brier']['resolution'])} "
          f"+ uncertainty {_fmt(t1['brier']['uncertainty'])}) · AUROC {_fmt(t1['auroc'])} · "
          f"AUPRC(correct) {_fmt(t1['auprc_correct'])} (base {_fmt(t1['base_rate_correct'])}) · n {t1['n']}")
        e("")
        e("```")
        L.extend(reliability_ascii(t1["bins"]))
        e("```")
        e("")
        t2 = r["tier2_per_statement"]
        a = t2["all"]
        e("### C0.3 — Tier-2 per-statement raw-belief (DECISIVE)")
        e("")
        e(f"- statements {t2['n_statements']} ({t2['n_singletons']} singletons, {t2['n_multi']} multi-evidence)")
        e(f"- **AUROC {_fmt(a['auroc'])} · AUPRC(correct) {_fmt(a['auprc_correct'])}** "
          f"(base {_fmt(a['base_rate_correct'])}) · ECE {_fmt(a['ece'])} · Brier {_fmt(a['brier']['brier'])} "
          f"(resolution {_fmt(a['brier']['resolution'])})")
        e(f"- saturation check: mean belief when correct {_fmt(a['mean_belief_when_correct'])} vs "
          f"when incorrect {_fmt(a['mean_belief_when_incorrect'])}")
        for sub in ("singletons", "multi_evidence", "indra_prior_reference"):
            h = t2[sub]
            if h:
                e(f"  - {sub}: n {h['n']}, AUROC {_fmt(h['auroc'])}, AUPRC {_fmt(h['auprc_correct'])} "
                  f"(base {_fmt(h['base_rate_correct'])}), ECE {_fmt(h['ece'])}")
        e("")
        e("```")
        L.extend(reliability_ascii(t2["bins"]))
        e("```")
        e("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/benchmark/eval_curation_v1.jsonl")
    ap.add_argument("--run", action="append", default=[], help="run output jsonl (repeatable)")
    ap.add_argument("--name", action="append", default=[], help="display name per --run (repeatable)")
    ap.add_argument("--out", default="data/results/calibration_stage0.md")
    ap.add_argument("--json", default="data/results/calibration_stage0.json")
    args = ap.parse_args()

    if not args.run:
        args.run = ["data/results/eval_curation_v1_medpsy.jsonl",
                    "data/results/eval_curation_v1_gemma.jsonl"]
        args.name = ["MedPsy-4B", "gemma-26B"]
    names = args.name + [f"run{i}" for i in range(len(args.name), len(args.run))]

    gold = load_jsonl(ROOT / args.gold)
    by_pair, by_sh = build_gold_index(gold)

    results = []
    for path, name in zip(args.run, names):
        rows = load_jsonl(ROOT / path)
        joined, pn, miss = join_model(rows, by_pair, by_sh)
        results.append(analyze(name, joined, pn, miss))

    md = render_md(results, args)
    (ROOT / args.out).write_text(md)
    (ROOT / args.json).write_text(json.dumps(results, indent=2, default=float))
    print(md)
    print(f"\nWrote {args.out} and {args.json}")


if __name__ == "__main__":
    main()
