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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.curation import is_gold_correct  # noqa: E402
from indra_belief.metrics import confusion_pr, ece  # noqa: E402

MASK = (1 << 64) - 1

# Bootstrap/permutation conventions, shared with calibration_ship_gate.bootstrap_errf1.
N_BOOT = 2000
N_PERM = 2000
BOOT_SEED = 0
# medpsy-4B Arm A run-to-run err-F1 spread observed at n=60 (entity-reading pilot).
# A between-model ΔerrF1 smaller than this is below the *small-sample* single-model
# noise band — which is precisely why model comparisons are made at n=1606, where the
# paired CI/permutation test below resolves effects the n=60 spread would have buried.
NOISE_FLOOR = 0.154


def _errf1_from_pairs(gold_err, pred_err) -> float:
    """Error-detection F1 over arrays of (gold_error, pred_error) booleans.

    Positive class = curator-flagged INCORRECT. Same definition confusion_pr
    consumes in model_block (ed_pairs), so the POINT estimate computed over the
    full keyset equals the f1 already emitted there.
    """
    return confusion_pr(list(zip([bool(x) for x in gold_err],
                                 [bool(x) for x in pred_err])))["f1"]


def bootstrap_errf1(gold_err, pred_err_a, pred_err_b,
                    n_boot=N_BOOT, seed=BOOT_SEED) -> dict:
    """Paired percentile bootstrap over the SHARED keyset for err-F1(A), err-F1(B),
    and ΔerrF1 = B − A.

    Mirrors calibration_ship_gate.bootstrap_errf1's structure but over the
    verdict-grain error-detection pairs: each statement carries one gold-error
    label and one pred-error per model; a resample picks statements (paired, so
    A and B see the same draw) and recomputes all three. Percentiles are 2.5/97.5.
    The POINT estimates returned are the full-keyset values (unchanged).
    """
    ge = np.asarray(gold_err, bool)
    pa = np.asarray(pred_err_a, bool)
    pb = np.asarray(pred_err_b, bool)
    n = len(ge)
    rng = np.random.default_rng(seed)
    f_a = np.empty(n_boot)
    f_b = np.empty(n_boot)
    f_d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ge_b = ge[idx]
        fa = _errf1_from_pairs(ge_b, pa[idx])
        fb = _errf1_from_pairs(ge_b, pb[idx])
        f_a[i] = fa
        f_b[i] = fb
        f_d[i] = fb - fa

    def ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    return {
        "f1_a": _errf1_from_pairs(ge, pa),
        "f1_b": _errf1_from_pairs(ge, pb),
        "delta": _errf1_from_pairs(ge, pb) - _errf1_from_pairs(ge, pa),
        "ci_a": ci(f_a),
        "ci_b": ci(f_b),
        "ci_delta": ci(f_d),
        "n": n,
    }


def permutation_errf1(gold_err, pred_err_a, pred_err_b,
                      n_perm=N_PERM, seed=BOOT_SEED) -> float:
    """Paired permutation p-value on |ΔerrF1| over the shared keyset.

    On each permutation, randomly swap the A/B pred-error labels per statement
    (exchangeability under H0: the two models are interchangeable), recompute
    |ΔerrF1| = |errF1(B') − errF1(A')|, and count the permuted |Δ| that meet or
    exceed the observed |Δ|.

    The returned p is (hits + 1) / (n_perm + 1), matching
    frontier_paired_stats.paired_permutation_errf1: the plus-one prevents a
    finite Monte Carlo run from reporting an impossible p-value of zero (the
    observed split is itself one of the exchangeable arrangements). p therefore
    lies in [1 / (n_perm + 1), 1] — 1/(n_perm + 1) is the minimum ATTAINABLE
    p, not evidence of a smaller one, and p == 1.0 exactly when every
    permutation ties or beats the observed |Δ|.
    """
    ge = np.asarray(gold_err, bool)
    pa = np.asarray(pred_err_a, bool)
    pb = np.asarray(pred_err_b, bool)
    n = len(ge)
    obs = abs(_errf1_from_pairs(ge, pb) - _errf1_from_pairs(ge, pa))
    rng = np.random.default_rng(seed)
    ge_dummy = ge  # gold labels are fixed under the swap
    hits = 0
    for _ in range(n_perm):
        swap = rng.integers(0, 2, n).astype(bool)
        pa_p = np.where(swap, pb, pa)
        pb_p = np.where(swap, pa, pb)
        d = abs(_errf1_from_pairs(ge_dummy, pb_p) - _errf1_from_pairs(ge_dummy, pa_p))
        if d >= obs:
            hits += 1
    return (hits + 1) / (n_perm + 1)


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


# ---- gold-join trio: build_gold_index / gold_for / join_model ----------------
# DELIBERATE, leaner variant — kept DISTINCT from scripts/calibration_stage0.py's
# same-named trio; the two are NOT to be unified. This trio is the stable public
# API imported by analyze_external_gold, bootstrap_precision, convergence_report,
# learning_curve_v2, demo_simulate_more_data (build_gold_index + join_model) and
# frontier_table (build_gold_index + gold_for), so it must NOT adopt stage0's
# collapsing / raising / permissive-fallback behavior.
#
# Three intentional call-site divergences from calibration_stage0.py's trio:
#   1. build_gold_index does NO multi-curator collapse — last-write-wins into
#      by_pair, raw rows kept in by_sh (stage0 collapses via any-incorrect-wins).
#   2. gold_for uses a STRICT source-only fallback: fire only when exactly one
#      gold row carries the source_hash (len(cand) == 1). stage0 is permissive
#      (fires whenever every candidate agrees on truth class, len(classes) == 1).
#   3. join_model does NO pair-dedup (stage0 dedups and raises on conflicting
#      scored verdicts for a duplicated pair).
#
# The two build_gold_index collapses were MEASURED byte-equal on the current
# golds (collapse_diff_pairs=0 on eval_curation_v1 n=1606 + external_curator_gold
# _v1 n=578); that split is a latent-drift guard, not an active mismatch. The
# gold_for source-only fallback rule, by contrast, is NOT proven equal (its
# resolution differs on those golds' source_hash sets) — a further reason not to
# force the merge.
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
    stmt_hash_hex = scored.get("stmt_hash")
    mh = int(stmt_hash_hex, 16) if stmt_hash_hex else None
    if mh is not None and (mh, sh) in by_pair:
        return by_pair[(mh, sh)]
    # Fallback when the statement hash is absent/skewed: accept a source_hash
    # match ONLY when it is unambiguous (exactly one gold row carries it);
    # anything else returns None rather than risk a wrong join.
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
    # Rows with no score are EXCLUDED, not imputed at 0.5. 0.5 is off the
    # six-cell grid, so imputing it fabricates a measurement for a row the model
    # never answered — and puts it exactly at the neutral point ECE is most
    # sensitive to. An absent measurement stays absent.
    cal_pairs = [(s["score"], is_gold_correct(g["tag"]))
                 for g, s in joined if s.get("score") is not None]
    n_unscored = len(joined) - len(cal_pairs)
    cal = ece(cal_pairs)
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
        "ed": ed, "ece": cal, "ece_n": len(cal_pairs), "n_unscored_excluded": n_unscored,
        "by_tag": dict(by_tag), "by_type": dict(by_type),
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
    ap.add_argument("--title", default="human-curation eval (eval_curation_v1)",
                    help="report subtitle; set when reusing on a different gold set")
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

    # Paired err-F1 inference over the SAME shared keyset McNemar uses.
    # positive class = curator-flagged INCORRECT; pred = verdict == 'incorrect'.
    gold_err = [not is_gold_correct(by_pair[k]["tag"]) for k in shared]
    a_perr = [Akv[k]["verdict"] == "incorrect" for k in shared]
    b_perr = [Bkv[k]["verdict"] == "incorrect" for k in shared]
    errf1_boot = (bootstrap_errf1(gold_err, a_perr, b_perr) if shared else None)
    errf1_perm = (permutation_errf1(gold_err, a_perr, b_perr) if shared else None)
    # Same name/semantics as frontier_paired_stats' `minimum_attainable_p`: with the
    # (hits+1)/(n_perm+1) correction, p can never go below this, so a p sitting AT it
    # is floored by the Monte Carlo budget, not measured to be smaller.
    p_floor = 1 / (N_PERM + 1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as out:
        n_gc = sum(1 for g in gold if is_gold_correct(g["tag"]))
        emit(out, f"# {A['name']} vs {B['name']} — {args.title}\n")
        emit(out, f"Human gold: {len(gold)} pairs ({n_gc} correct / {len(gold) - n_gc} incorrect).\n")

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
        # LEAD: paired ΔerrF1 95% CI + permutation p over the shared keyset.
        if errf1_boot is not None:
            d = errf1_boot["delta"]; cd = errf1_boot["ci_delta"]
            sig = "significant" if errf1_perm < 0.05 else "not significant"
            direction = (f"{B['name']}" if d > 0 else f"{A['name']}") if d != 0 else "tie"
            emit(out, f"**ΔerrF1 ({B['name']} − {A['name']}) = {d:+.3f}  "
                      f"[95% CI {cd[0]:+.3f}, {cd[1]:+.3f}]  "
                      f"(paired bootstrap, {N_BOOT} resamples, seed {BOOT_SEED}; "
                      f"n={errf1_boot['n']} shared)**")
            emit(out, f"**Permutation p (err-F1, two-sided, {N_PERM} perms) = {errf1_perm:.4f}  "
                      f"({sig} at α=0.05; favors {direction})** — this is the PRIMARY paired test. "
                      f"minimum attainable p = {p_floor:.4f}")
            vs_floor = ("below" if abs(d) < NOISE_FLOOR else "above")
            emit(out, f"_Context: |ΔerrF1| = {abs(d):.3f} is {vs_floor} the n=60 single-model "
                      f"run-to-run spread (NOISE_FLOOR = {NOISE_FLOOR:.3f}, medpsy-4B Arm A). "
                      f"A gap can be {vs_floor} that small-sample floor yet still {sig} here, "
                      f"because the paired test is at n={errf1_boot['n']} — situate the effect "
                      f"against the CI, not the n=60 band._")
            emit(out, "")
        emit(out, "| model | precision | recall | F1 | F1 95% CI (paired n) | TP | FP | FN | TN |")
        emit(out, "|---|---|---|---|---|---|---|---|---|")
        for M, ci_key in ((A, "ci_a"), (B, "ci_b")):
            e = M["ed"]
            if errf1_boot is not None:
                lo, hi = errf1_boot[ci_key]
                ci_cell = f"[{lo:.3f}, {hi:.3f}]"
            else:
                ci_cell = "—"
            emit(out, f"| {M['name']} | {e['p']:.3f} | {e['r']:.3f} | **{e['f1']:.3f}** | {ci_cell} | "
                      f"{e['tp']} | {e['fp']} | {e['fn']} | {e['tn']} |")
        emit(out, "\n_Recall = fraction of real errors caught; precision = of flagged, how many were truly wrong._")
        emit(out, "_F1 point estimate is over each model's full join; the 95% CI (and the "
                  "ΔerrF1/permutation test above) are over the shared paired keyset, so the F1 "
                  "the CI brackets is the paired-subset value, which can differ slightly from the "
                  "full-join point._\n")

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

        emit(out, "## Paired comparison (McNemar on accuracy — SECONDARY)")
        emit(out, "_Lead with the paired ΔerrF1 CI + permutation p in the Error-detection "
                  "section above; this McNemar-on-accuracy test is reported as the secondary "
                  "paired check._")
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
    if errf1_boot is not None:
        cd = errf1_boot["ci_delta"]
        print(f"ΔerrF1 ({B['name']}−{A['name']})={errf1_boot['delta']:+.3f} "
              f"[95% CI {cd[0]:+.3f},{cd[1]:+.3f}]  permutation p={errf1_perm:.4f}  "
              f"(PRIMARY; n={errf1_boot['n']} shared; minimum attainable p={p_floor:.4f})")
    print(f"McNemar p={p:.4f}  (b={a_only} {A['name']}-only, c={b_only} {B['name']}-only)  [secondary]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
