"""MedPsy-vs-gemma head-to-head on the balanced human-curation eval set.

Joins two blind monolithic scoring runs (run_rasmachine_monolithic.py output)
back to eval_curation_v1.jsonl gold on the (matches_hash, source_hash) PAIR —
source_hash alone is not unique, and joining on it would silently mismatch (the
version-skew lesson, applied to the in-corpus case). Reuses the canonical libs
(indra_belief.metrics, indra_belief.curation) so these numbers share one
definition with every other eval.

The headline is ERROR DETECTION (positive class = the curator-flagged incorrect
extraction), not accuracy: gold is balanced 1:1 here, but accuracy still rewards
a lenient acceptor on the easy class. Catching the wrong extractions is the job.

    PYTHONPATH=src python scripts/eval_curation_compare.py \
        --gold data/benchmark/eval_curation_v1.jsonl \
        --a data/results/eval_curation_v1_medpsy.jsonl --a-name MedPsy-4B \
        --b data/results/eval_curation_v1_gemma.jsonl  --b-name gemma-26B \
        --out data/results/eval_curation_v1_compare.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.curation import is_gold_correct  # noqa: E402
from indra_belief.metrics import confusion_pr, ece  # noqa: E402

MASK = (1 << 64) - 1


def umask(x) -> int:
    return int(x) & MASK


def load_jsonl(p: str | Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value over discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # exact binomial tail at p=0.5, two-sided
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def build_gold_index(gold_rows: list[dict]):
    by_pair: dict[tuple[int, int], dict] = {}
    by_sh: dict[int, list[dict]] = defaultdict(list)
    for r in gold_rows:
        mh = umask(r["matches_hash"]); sh = umask(r["source_hash"])
        by_pair[(mh, sh)] = r
        by_sh[sh].append(r)
    return by_pair, by_sh


def gold_for(scored: dict, by_pair, by_sh) -> dict | None:
    sh = umask(scored["source_hash"])
    sh_hex = scored.get("stmt_hash")
    mh = int(sh_hex, 16) if sh_hex else None
    if mh is not None and (mh, sh) in by_pair:
        return by_pair[(mh, sh)]
    cand = by_sh.get(sh, [])
    return cand[0] if len(cand) == 1 else None


def join_model(scored_rows, by_pair, by_sh):
    """Return list of (gold_row, scored_row) plus parse/miss stats."""
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


def model_block(name: str, joined: list[tuple[dict, dict]]) -> dict:
    """All single-model metrics over joined (gold, scored) rows."""
    # accuracy
    acc_hits = sum(1 for g, s in joined if (s["verdict"] == "correct") == is_gold_correct(g["tag"]))
    n = len(joined)
    # error-detection: positive = incorrect
    ed_pairs = [(not is_gold_correct(g["tag"]), s["verdict"] == "incorrect") for g, s in joined]
    ed = confusion_pr(ed_pairs)
    # calibration: score is P(correct); is_correct = gold-correct
    cal = ece([(s.get("score") if s.get("score") is not None else 0.5, is_gold_correct(g["tag"]))
               for g, s in joined])
    # per-tag correct-call rate
    by_tag: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [right, total]
    for g, s in joined:
        right = (s["verdict"] == "correct") == is_gold_correct(g["tag"])
        by_tag[g["tag"]][0] += int(right); by_tag[g["tag"]][1] += 1
    # per-stmt_type accuracy
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for g, s in joined:
        right = (s["verdict"] == "correct") == is_gold_correct(g["tag"])
        by_type[g["stmt_type"]][0] += int(right); by_type[g["stmt_type"]][1] += 1
    lo, hi = wilson_ci(acc_hits, n)
    return {
        "name": name, "n": n, "acc": acc_hits / n if n else 0, "acc_ci": (lo, hi),
        "ed": ed, "ece": cal, "by_tag": dict(by_tag), "by_type": dict(by_type),
    }


def emit(out, lines: str = "") -> None:
    out.write(lines + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", default=str(ROOT / "data/benchmark/eval_curation_v1.jsonl"))
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--out", default=str(ROOT / "data/results/eval_curation_v1_compare.md"))
    args = ap.parse_args()

    gold = load_jsonl(args.gold)
    by_pair, by_sh = build_gold_index(gold)
    A_rows, B_rows = load_jsonl(args.a), load_jsonl(args.b)
    A_join, A_null, A_miss = join_model(A_rows, by_pair, by_sh)
    B_join, B_null, B_miss = join_model(B_rows, by_pair, by_sh)
    A = model_block(args.a_name, A_join)
    B = model_block(args.b_name, B_join)

    # paired comparison over the intersection (both parsed) by gold pair key
    def keyset(joined):
        return {(umask(g["matches_hash"]), umask(g["source_hash"])): s for g, s in joined}
    Akv, Bkv = keyset(A_join), keyset(B_join)
    shared = sorted(set(Akv) & set(Bkv))
    a_right = b_right = both_r = both_w = 0
    a_only = b_only = 0  # discordant: a right & b wrong / vice versa
    acbi = []  # A correct, B incorrect (verdict disagreements), with gold
    aibc = []
    for k in shared:
        g = by_pair[k]
        gc = is_gold_correct(g["tag"])
        ar = (Akv[k]["verdict"] == "correct") == gc
        br = (Bkv[k]["verdict"] == "correct") == gc
        a_right += ar; b_right += br
        if ar and br: both_r += 1
        elif not ar and not br: both_w += 1
        elif ar and not br: a_only += 1
        else: b_only += 1
        # verdict-level disagreement (who calls what)
        if Akv[k]["verdict"] != Bkv[k]["verdict"]:
            rec = {"subj": g["subject"], "obj": g["object"], "type": g["stmt_type"],
                   "tag": g["tag"], "gold": g["gold"],
                   "a": Akv[k]["verdict"], "b": Bkv[k]["verdict"]}
            if Akv[k]["verdict"] == "correct":
                acbi.append(rec)
            else:
                aibc.append(rec)
    p = mcnemar_p(a_only, b_only)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as out:
        emit(out, f"# {A['name']} vs {B['name']} — human-curation eval (eval_curation_v1)\n")
        emit(out, f"Balanced 1:1 human gold, fresh + de-contaminated. Gold pairs: {len(gold)}.\n")

        emit(out, "## Coverage")
        emit(out, f"- {A['name']}: joined {A['n']}  (parse-null {A_null}, unmatched {A_miss})")
        emit(out, f"- {B['name']}: joined {B['n']}  (parse-null {B_null}, unmatched {B_miss})")
        emit(out, f"- paired (both parsed): {len(shared)}\n")

        emit(out, "## Headline — accuracy (verdict == gold)")
        for M in (A, B):
            lo, hi = M["acc_ci"]
            emit(out, f"- **{M['name']}: {M['acc']:.1%}**  (95% CI {lo:.1%}–{hi:.1%}, n={M['n']})")
        emit(out, "")

        emit(out, "## Error detection (positive class = curator-flagged INCORRECT)")
        emit(out, "| model | precision | recall | F1 | TP | FP | FN | TN |")
        emit(out, "|---|---|---|---|---|---|---|---|")
        for M in (A, B):
            e = M["ed"]
            emit(out, f"| {M['name']} | {e['p']:.3f} | {e['r']:.3f} | **{e['f1']:.3f}** | "
                      f"{e['tp']} | {e['fp']} | {e['fn']} | {e['tn']} |")
        emit(out, "\n_Recall = fraction of real errors caught; precision = of flagged, how many were truly wrong._\n")

        emit(out, "## Calibration (ECE, 8-bin)")
        for M in (A, B):
            emit(out, f"- {M['name']}: **{M['ece']:.3f}**")
        emit(out, "")

        emit(out, "## Per-gold-tag correct-call rate (where each model fails)")
        tags = sorted(set(A["by_tag"]) | set(B["by_tag"]),
                      key=lambda t: -(A["by_tag"].get(t, [0, 0])[1]))
        emit(out, f"| gold tag | n | {A['name']} | {B['name']} |")
        emit(out, "|---|---|---|---|")
        for t in tags:
            ar, an = A["by_tag"].get(t, [0, 0])
            br, bn = B["by_tag"].get(t, [0, 0])
            nn = max(an, bn)
            emit(out, f"| {t} | {nn} | {ar}/{an} ({ar/an:.0%}) | {br}/{bn} ({br/bn:.0%}) |"
                 if an and bn else f"| {t} | {nn} | {ar}/{an} | {br}/{bn} |")
        emit(out, "")

        emit(out, "## Per-stmt_type accuracy")
        types = sorted(set(A["by_type"]) | set(B["by_type"]),
                       key=lambda t: -(A["by_type"].get(t, [0, 0])[1]))
        emit(out, f"| stmt_type | n | {A['name']} | {B['name']} |")
        emit(out, "|---|---|---|---|")
        for t in types:
            ar, an = A["by_type"].get(t, [0, 0])
            br, bn = B["by_type"].get(t, [0, 0])
            emit(out, f"| {t} | {max(an,bn)} | {ar/an:.0%} | {br/bn:.0%} |"
                 if an and bn else f"| {t} | {max(an,bn)} | — | — |")
        emit(out, "")

        emit(out, "## Paired comparison (McNemar)")
        emit(out, f"- both right: {both_r}   both wrong: {both_w}")
        emit(out, f"- {A['name']} right & {B['name']} wrong (b): **{a_only}**")
        emit(out, f"- {B['name']} right & {A['name']} wrong (c): **{b_only}**")
        emit(out, f"- McNemar two-sided exact p = **{p:.4f}**  "
                  f"({'significant' if p < 0.05 else 'not significant'} at α=0.05)")
        winner = A["name"] if a_only > b_only else (B["name"] if b_only > a_only else "tie")
        emit(out, f"- direction: {winner}\n")

        emit(out, f"## Verdict disagreements ({len(acbi)+len(aibc)} pairs)")
        emit(out, f"- {A['name']}=correct, {B['name']}=incorrect: {len(acbi)}")
        emit(out, f"- {A['name']}=incorrect, {B['name']}=correct: {len(aibc)}")
        for label, recs in ((f"{A['name']}✓ / {B['name']}✗", acbi),
                            (f"{A['name']}✗ / {B['name']}✓", aibc)):
            if not recs:
                continue
            emit(out, f"\n### {label} (gold tag shown)")
            emit(out, "| subj | type | obj | gold | tag |")
            emit(out, "|---|---|---|---|---|")
            for r in sorted(recs, key=lambda r: r["gold"])[:40]:
                emit(out, f"| {r['subj']} | {r['type']} | {r['obj']} | {r['gold']} | {r['tag']} |")

    print(f"wrote {out_path}")
    # console headline
    print(f"\n{A['name']}: acc {A['acc']:.1%}  error-F1 {A['ed']['f1']:.3f}  ECE {A['ece']:.3f}")
    print(f"{B['name']}: acc {B['acc']:.1%}  error-F1 {B['ed']['f1']:.3f}  ECE {B['ece']:.3f}")
    print(f"McNemar p={p:.4f}  (b={a_only} {A['name']}-only, c={b_only} {B['name']}-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
