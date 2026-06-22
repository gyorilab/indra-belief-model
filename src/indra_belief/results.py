"""Run-output enrichment + the bucket taxonomy — the single source of truth.

Turns a raw monolithic scoring run (``data/results/<run>.jsonl``) into the
consumable export the viewer *and* collaborators read, without any one-off
script in the loop:

  per_evidence.jsonl  one (statement, evidence) row — RasMachine belief vs. our
                      per-evidence score, the model's verdict/confidence/reasoning,
                      the joined evidence sentence + INDRA ids, and the bucket label
  per_statement.json  one row per ``stmt_hash`` — belief vs. aggregate (mean +
                      noisy-OR), evidence depth, verdict tally, dominant bucket
  export_meta.json    provenance + counts + join/reasoning quality

The bucket taxonomy (``classify`` / ``split_preview`` / ``META`` / ``ORDER``)
lives here as the single definition the export uses.

Use it as a library (the runner calls :func:`write_run_export` at the end of a
run) or as a CLI for an existing run::

    python -m indra_belief.results data/results/<run>.jsonl
    python -m indra_belief.results <run>.jsonl --out data/exports/<name>
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any

from indra_belief.curation import aggregate_gold, is_gold_correct
from indra_belief.corpus.cost import price_basis, token_cost_usd
from indra_belief.model_client import canonical_model_name
from indra_belief.model_meta import model_size
from indra_belief.calibration_constants import calibration_for
from indra_belief.metrics import (
    BINS_8,
    auroc,
    auprc,
    brier_murphy,
    confusion_metrics,
    ece,
    reliability_bins,
)
from indra_belief.noise_model import RECALIBRATED_PRIORS
from indra_belief.statement_belief import statement_belief

DEFAULT_CORPUS = "data/corpora/latest_statements_rasmachine.json"

# ── bucket taxonomy ─────────────────────────────────────────────────────────
# Strict partition (each row in exactly one bucket) under top-to-bottom
# precedence: telemetry > schema-shape > verdict-based.

HALLUC = re.compile(
    r"(does not (mention|name|state|describe)|not mentioned|"
    r"doesn.t (mention|name)|no mention of|fails to mention)",
    re.I,
)
HEDGE = re.compile(
    r"\b(may|might|suggests?|potentially|appears? to|could|"
    r"seem(s|ed)? to|imply|implies|likely|hypothesi[sz]e)\b",
    re.I,
)
# Evidence appears in two prompt templates; some rows (no_evidence) have neither.
EV_COLON = re.compile(r'The evidence is:\s*"(.*?)"\s*(?:\n|$)', re.S)
EV_TICK = re.compile(r'The evidence is\s*`"(.*?)"`', re.S)

# Captured evidence is accepted only if its length is within tolerance of the
# row's recorded ``text_len`` — this rejects over-captures where the regex would
# swallow the trailing reasoning. Tolerance = max(text_len + slack, text_len * factor).
EV_LEN_SLACK = 30  # absolute char slack (covers quoting/whitespace on short evidence)
EV_LEN_FACTOR = 1.3  # relative slack (covers longer evidence proportionally)
# Below this many characters an evidence string is treated as a stub / DB token
# (``placeholder_text``) rather than a real sentence.
PLACEHOLDER_MAX_LEN = 30

# bucket -> (group-key, remedy/explanation phrase)
META: dict[str, tuple[str, str]] = {
    "semantic_correct": ("sem", "model affirms the claim"),
    "semantic_incorrect": ("sem", "genuine polarity / mechanism / direction mismatch"),
    "hedged_evidence": ("sem", "policy: hedged claims may be valid-but-uncertain"),
    "reader_hallucination": ("rdr", "NLP reader extracted an entity not in the evidence text"),
    "no_evidence": ("sch", "source-DB record without an evidence sentence"),
    "incomplete_claim": ("sch", "ActiveForm is unary; rendered as ternary SUBJ [REL] ?"),
    "placeholder_text": ("sch", "stub strings / short DB tokens"),
    "row_error": ("tel", "transport/timeout/runner-side failure"),
}
ORDER = [
    "semantic_correct", "semantic_incorrect", "hedged_evidence", "reader_hallucination",
    "no_evidence", "incomplete_claim", "placeholder_text", "row_error",
]
GROUP_NAME = {"sem": "semantic", "rdr": "reader-artifact", "sch": "schema-artifact", "tel": "telemetry"}


def split_preview(p: str | None, text_len: int) -> tuple[str, str]:
    """Return ``(evidence_sentence, reasoning)`` from a ``raw_text_preview``.

    The evidence capture is validated against ``text_len`` so over-captures
    (empty-evidence rows where the regex would swallow the reasoning) are
    rejected and yield ``""`` for genuine no-evidence rows.
    """
    if not p:
        return "", ""
    ev, reasoning = "", p
    m = EV_COLON.search(p) or EV_TICK.search(p)
    if m:
        reasoning = p[m.end():]
        cand = m.group(1).strip()
        if text_len and len(cand) <= max(text_len + EV_LEN_SLACK, int(text_len * EV_LEN_FACTOR)):
            ev = cand
    if not text_len:
        ev = ""
    return ev, reasoning


def classify(d: dict[str, Any], ev: str, reasoning: str) -> str:
    """Assign a row to exactly one bucket (see module docstring for predicates)."""
    if d.get("error") is not None or d.get("verdict") is None:
        return "row_error"
    tl = d.get("text_len") or 0
    if tl == 0:
        return "no_evidence"
    if tl < PLACEHOLDER_MAX_LEN:
        return "placeholder_text"
    if d.get("stmt_type") == "ActiveForm" and d.get("object") in (None, "?", ""):
        return "incomplete_claim"
    v = d.get("verdict")
    if v == "correct":
        return "semantic_correct"
    if v == "incorrect":
        if HALLUC.search(reasoning):
            return "reader_hallucination"
        if ev and HEDGE.search(ev):
            return "hedged_evidence"
        return "semantic_incorrect"
    return "row_error"


def corpus_evidence_text(path: str = DEFAULT_CORPUS) -> dict[tuple[int, int], str]:
    """``(stmt_i, evidence_i) -> evidence sentence`` from the corpus (cached).

    Positional index (validated byte-exact against the run); the empty-string
    marker reflects only genuine no-evidence records.
    """
    global _CORPUS_TEXT
    if _CORPUS_TEXT is None:
        with open(path) as f:
            corpus = json.load(f)
        m: dict[tuple[int, int], str] = {}
        for si, stmt in enumerate(corpus):
            for ei, ev in enumerate(stmt.get("evidence") or []):
                m[(si, ei)] = ev.get("text") or ""
        _CORPUS_TEXT = m
    return _CORPUS_TEXT


_CORPUS_TEXT: dict[tuple[int, int], str] | None = None


# ── enrichment ──────────────────────────────────────────────────────────────

def _r3(x: Any) -> float | None:
    return round(x, 3) if isinstance(x, (int, float)) else None


def call_log_cost(call_log: list[dict]) -> dict:
    """Observed USD + token totals + cost availability for one row's call_log.

    Single pass: token totals AND USD are accumulated together so they can never
    diverge. Cost is UNAVAILABLE (cost_usd=None, cost_status="unavailable") iff ANY
    call used a model_id with no price at all (list or estimate; including model_id
    None / "unknown") — we never report a partial/fabricated $0 for an unpriced
    model. Otherwise cost_usd is the sum of token_cost_usd over every call. When any
    priced call used a Bedrock-GROUNDED ESTIMATE (a local/self-hosted model, no
    observed list price) the status is "estimated" rather than "known". Empty/
    missing call_log -> $0.00 over 0 tokens with cost_status="known" (a no-LLM row,
    e.g. Tier-0 no_text / Tier-1 auto-reject, genuinely costs $0).

    FLOOR caveat: when a PRICED call's prompt_tokens is the unreported sentinel
    (-1, model_client.ModelResponse default when a backend omits usage.prompt_tokens)
    it is clamped to 0 — so input_tokens and the input portion of cost_usd are a
    LOWER BOUND, while cost_status stays "known". This is latent in practice (the
    Bedrock OpenAI-compat mantle reports usage.prompt_tokens), but a priced backend
    that omits input usage would understate cost without flagging it. Output-only
    cost is still exact.
    """
    calls = call_log or []
    in_tok = out_tok = 0
    cost = 0.0
    all_known = True
    any_estimate = False
    seen_models: set[str] = set()
    for c in calls:
        mid = c.get("model_id")
        if mid:
            seen_models.add(mid)
        pt = c.get("prompt_tokens")
        ot = c.get("out_tokens")
        in_tok += pt if isinstance(pt, (int, float)) and pt > 0 else 0
        out_tok += ot if isinstance(ot, (int, float)) and ot > 0 else 0
        basis = price_basis(mid) if mid else None
        if basis is None:
            all_known = False
        else:
            if basis == "estimate":
                any_estimate = True
            # same clamping as the totals via token_cost_usd -> _nonnegative_tokens
            cost += token_cost_usd(mid, pt, ot, on_unknown="zero")
    if not all_known:
        # `cost` accumulated above is discarded here — we never expose a partial
        # USD for a row that touched an unverified model.
        return {
            "cost_usd": None, "cost_status": "unavailable",
            "input_tokens": in_tok, "output_tokens": out_tok,
            "n_calls": len(calls), "models": sorted(seen_models),
        }
    # "estimated": at least one priced call used a Bedrock-grounded estimate (a
    # local/self-hosted model) rather than an observed list price.
    return {
        "cost_usd": round(cost, 6), "cost_status": "estimated" if any_estimate else "known",
        "input_tokens": in_tok, "output_tokens": out_tok,
        "n_calls": len(calls), "models": sorted(seen_models),
    }


# Free chain-of-thought can run long (gemma plaintext ~1.5k chars); clip it for
# the per-evidence export the way `reasoning` is clipped, but keep the full length
# so the viewer can show a "truncated" affordance. The status / tokens /
# committed-justification fields are small and always kept.
_FREE_COT_CLIP = 4000


def compact_reasoning_trace(rt: Any) -> dict | None:
    """Project a raw call_log reasoning_trace into the per-evidence export shape.

    Keeps the small, always-useful fields (status, reasoning_tokens, provenance,
    committed support/objection) verbatim and clips the free CoT. Returns None
    for legacy rows (no trace) so the viewer can distinguish "no trace recorded"
    from a present-but-empty trace."""
    if not isinstance(rt, dict):
        return None
    cot = rt.get("free_cot") or ""
    cj = rt.get("committed_justification") or {}
    return {
        "status": rt.get("status"),
        "reasoning_tokens": rt.get("reasoning_tokens"),
        "provider_source": rt.get("provider_source"),
        "backend": rt.get("backend"),
        "model_id": rt.get("model_id"),
        "finish_reason": rt.get("finish_reason"),
        "free_cot": cot[:_FREE_COT_CLIP],
        "free_cot_chars": len(cot),
        "committed_justification": {
            "support": cj.get("support"),
            "objection": cj.get("objection"),
            "source": cj.get("source"),
        },
    }


def _count_trace_status(per_ev: list[dict]) -> dict[str, int]:
    """Histogram of reasoning_trace.status across the export (legacy rows with no
    trace bucket under 'no_trace')."""
    out: dict[str, int] = {}
    for r in per_ev:
        rt = r.get("reasoning_trace")
        key = rt.get("status") if isinstance(rt, dict) else "no_trace"
        out[key or "no_trace"] = out.get(key or "no_trace", 0) + 1
    return out


def _read_run_meta(run_path: str) -> dict[str, Any]:
    """Best-effort read of the run's sibling ``<run>.meta.json`` (run_id, model)."""
    meta_path = re.sub(r"\.jsonl$", ".meta.json", run_path)
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return {}


_HASH_MASK = (1 << 64) - 1


def _ukey(x) -> int | None:
    """Unsigned 64-bit int key — the source_hash join used everywhere (run rows,
    eval gold, curation.ts curationKey all resolve to the same integer)."""
    try:
        return int(x) & _HASH_MASK
    except (ValueError, TypeError):
        return None


def load_gold_map(gold_path: str) -> dict[int, dict]:
    """Build ``source_hash -> GoldVerdict`` from a curation JSONL so an export can
    BAKE its own gold in (gold then travels with the run and switches when you
    switch runs — no global hardcoded curations file).

    Keyed on SOURCE_HASH, not the (matches_hash, source_hash) pair: a run row's
    source_hash is authoritative (it IS the scored evidence), whereas the export's
    matches_hash can be a CorpusIndex fallback statement when subject/object
    didn't uniquely resolve — joining on the pair silently drops those rows.

    Accepts any curation row with ``source_hash`` + ``tag`` (and optional
    ``all_tags``/``curator``/``curator_note``/``text``): eval_curation_v1,
    belief_benchmark, rasmachine_curations all fit. Rows sharing a source_hash
    aggregate any-incorrect-wins (curation.aggregate_gold). Emits the viewer's
    GoldVerdict shape so goldForRow returns it verbatim."""
    groups: dict[int, list[dict]] = defaultdict(list)
    with open(gold_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sh = _ukey(r.get("source_hash"))
            if sh is None:
                continue
            groups[sh].append(r)

    gold: dict[int, dict] = {}
    for sh, rs in groups.items():
        tags: list[str] = []
        for x in rs:
            tags += x.get("all_tags") or ([x["tag"]] if x.get("tag") else [])
        verdict = aggregate_gold(tags)
        if verdict is None:
            continue
        notes = [n for x in rs
                 for n in [(x.get("curator_note") or x.get("text") or "").strip()] if n]
        gold[sh] = {
            "verdict": verdict,
            "n": len(rs),
            "tags": sorted(set(tags)),
            "curators": sorted({x.get("curator") for x in rs if x.get("curator")}),
            "notes": notes,
        }
    return gold


def _soft_calibration_block(model: str | None) -> dict:
    """Per-run soft-weight calibration (E5): the fitted triple that WOULD apply to
    this reader, baked so it travels with the run (per the per-run doctrine). When
    the reader has no fit, status is 'unavailable' WITH a reason — never an imputed
    zero. NOTE: the soft path is default-off; this records which calibration
    *applies* to the run, not that it was used to compute any belief here."""
    soft = calibration_for(model)
    if soft is None:
        return {
            "status": "unavailable",
            "model": model,
            "soft_weights": None,
            "reason": ("no model recorded for this run" if not model else
                       f"no fitted soft-weight calibration for reader {model!r} "
                       "(only gemma-4-26B / medpsy-4B are fitted)"),
        }
    return {"status": "available", "model": model, "soft_weights": soft}


METRICS_SCHEMA_VERSION = 2  # the metrics.json contract the viewer (C4/C5) pins
# v2: + tiers.stmt.verdict_err (error-detection confusion on the TIERED
#     verdict_statement, positive=ERROR) and tiers.stmt.stratified (per
#     stmt_type / source-count / evidence-count / bucket-group / reject-driver
#     residual). Purely additive — arms{hard,parametric,soft} unchanged + byte-exact.


def _metric_block(scores: list[float], labels: list[bool], tau: float) -> dict:
    """The stable per-arm metric unit C4/C5 read: {n, ece, auroc, auprc, brier,
    reliability, resolution, uncertainty, confusion{tp,fp,fn,tn}, bins[8]}.

    All math reuses src/indra_belief/metrics.py (the same definitions
    calibration_ship_gate.py goes through → served numbers cross-check exactly).
    `confusion` positive class = ERROR (pred_error = score < tau), matching the
    ship gate's err-F1 axis. `bins` is always exactly the 8 BINS_8 entries
    (unoccupied → n:0, mean_pred/empirical null) for a stable reliability x-axis.
    """
    b = brier_murphy(scores, labels)
    # err axis: positive = error (gold incorrect); pred_error = score < tau.
    conf = confusion_metrics([((not y), (s < tau)) for s, y in zip(scores, labels)])
    au = auroc(scores, labels)
    ap = auprc(scores, labels)
    return {
        "n": len(scores),
        "ece": ece(list(zip(scores, [bool(y) for y in labels]))),
        "auroc": None if au != au else au,   # NaN (single-class) → null
        "auprc": None if ap != ap else ap,
        "brier": b["brier"],
        "reliability": b["reliability"],
        "resolution": b["resolution"],
        "uncertainty": b["uncertainty"],
        "confusion": {"tp": conf["tp"], "fp": conf["fp"], "fn": conf["fn"], "tn": conf["tn"]},
        "bins": reliability_bins(scores, labels),
    }


def _verdict_err_confusion(rows: list[dict]) -> dict:
    """Error-detection confusion for the TIERED ``verdict_statement`` vs statement
    gold — the first-class statement heuristic the leaderboard reads.

    positive = ERROR. ``pred_error = verdict_statement != "correct"`` (review AND
    incorrect both count as flagged — the shipped review tier is an error-side
    escalation, not a clean pass). ``gold_error = statement gold is incorrect``
    (any-incorrect-wins). Long-key shape {n,tp,fp,fn,tn,accuracy,precision,recall,
    f1} so error precision/recall/F1 read straight off it. Distinct from the
    per-arm belief-threshold confusion inside ``_metric_block`` (belief < tau)."""
    pairs = [((not r["gold_correct"]), (r["verdict_statement"] != "correct")) for r in rows]
    return confusion_metrics(pairs)


def _stmt_stratum(rows: list[dict], tau: float) -> dict:
    """One statement-grain stratum: the verdict-driven error-detection confusion
    + the hard-arm belief calibration block, both over the same rows. Lets R7 read
    where error mass concentrates (verdict_err.f1) and where belief mis-calibrates
    (hard.ece) per stratum."""
    labels = [r["gold_correct"] for r in rows]
    return {
        "n": len(rows),
        "base_rate_correct": sum(labels) / len(labels),
        "verdict_err": _verdict_err_confusion(rows),
        "hard": _metric_block([r["hard"] for r in rows], labels, tau=tau),
    }


def _stratify(rows: list[dict], keyfn, tau: float) -> dict:
    """Group rows by ``keyfn`` (None keys dropped → not every statement carries
    every stratum dim) and emit a stratum block per group, key-sorted for a
    stable, byte-comparable JSON ordering."""
    groups: dict[Any, list[dict]] = defaultdict(list)
    for r in rows:
        k = keyfn(r)
        if k is None:
            continue
        groups[k].append(r)
    return {str(k): _stmt_stratum(v, tau)
            for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def build_run_metrics(
    per_ev: list[dict],
    stmt_agg: dict[str, dict],
    gold_map: dict[int, dict] | None,
    model: str | None,
    run_id: str | None,
    gold_path: str | None,
    *,
    tau: float = 0.5,
) -> dict:
    """The per-run calibration-product contract (E5) — `metrics.json`.

    Two tiers, keyed per run + per model:
      ev   (Tier-1) one realized arm — the per-evidence `our_score` vs gold.
      stmt (Tier-2) three arms — hard / parametric / soft belief vs statement gold.

    Gold travels with the run (baked `gold_map`, joined on source_hash). When no
    gold is baked, each tier is named-empty (`status: unavailable` + reason, no
    arms — never imputed zeros). When gold exists but the reader has no soft-weight
    fit, only the `soft` arm is named-empty; hard/parametric still render.

    Tier-2 mirrors the calibration arc EXACTLY so the served hard/soft ECE+AUROC
    cross-check calibration_ship_gate.py byte-for-byte: it re-builds a pair-index
    from gold_path (mirroring calibration_stage0.build_gold_index/gold_for),
    joins each evidence on (matches_hash, source_hash) with a unique-source_hash
    fallback, drops parse-null verdicts, groups statements by the gold's pa_hash,
    and rolls up with dedup OFF (the calibration arc does not de-dup). The frozen
    calibration_constants are the SAME values the ship gate re-fits (rounded to 3
    dp), so the soft arm matches within rounding (documented; hard/parametric are
    exact). Statement gold = any-incorrect-wins over its evidence tags."""
    soft = calibration_for(model)
    basis = {
        "bins": "BINS_8",
        "tau": tau,
        "join": ("Tier-1: source_hash (per-run baked gold). Tier-2: (matches_hash, "
                 "source_hash) pair with unique-source_hash fallback (mirrors "
                 "calibration_stage0.gold_for)."),
        "tier2_statement_key": ("gold pa_hash; statement gold = any-incorrect-wins; "
                                "belief rolled up with dedup OFF (matches the "
                                "calibration arc → cross-checks calibration_ship_gate)"),
        "soft_calibration": (
            {"status": "available", "w_correct": soft["w_correct"],
             "w_incorrect": soft["w_incorrect"], "variant": soft.get("variant", "guard")}
            if soft else {"status": "unavailable"}),
        "definitions": ("ece/BINS_8/confusion_metrics + auroc/auprc/brier_murphy/"
                        "reliability_bins all from src/indra_belief/metrics.py"),
        "confusion_axis": ("Tier-1: pred_correct = our_score >= tau (positive=correct). "
                           "Tier-2: pred_error = belief < tau (positive=ERROR), matching "
                           "calibration_ship_gate err-F1."),
        "soft_weights_note": ("soft arm uses the FROZEN calibration_constants (3-dp "
                              "rounded); calibration_ship_gate re-fits the same values, so "
                              "the soft arm matches it within rounding; hard/parametric exact."),
        "statement_verdict_err": ("tiers.stmt.verdict_err (+ per stratum): error-"
                                  "detection confusion on the TIERED verdict_statement, NOT "
                                  "belief<tau. positive=ERROR; pred_error = verdict_statement "
                                  "!= 'correct' (review AND incorrect flagged); gold_error = "
                                  "statement gold incorrect (any-incorrect-wins)."),
        "statement_strata": ("tiers.stmt.stratified dims: by_stmt_type, by_n_sources "
                             "(single/multi POST-gate distinct reads behind the belief), "
                             "by_n_evidence (single/multi RAW attached evidence, incl. "
                             "no_text/parse_fail — so a statement can be multi-evidence yet "
                             "single-source), by_dominant_bucket (grouped bucket), by_driver "
                             "(deterministic/llm/none reject driver). Each stratum = {n, "
                             "base_rate_correct, verdict_err, hard:<metric_block>}. Diagnostic "
                             "residual map; the shipped arms are unchanged."),
    }
    out: dict = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "run_id": run_id,
        "model": model,
        "generated_date": datetime.date.today().isoformat(),
        "metrics_basis": basis,
        "gold": ({"source": gold_path,
                  "covered": sum(1 for r in per_ev if r.get("gold")),
                  "total": len(per_ev)} if gold_map is not None else None),
        "tiers": {},
    }

    if gold_map is None:
        reason = "no gold baked for this run (pass --gold to enable calibration metrics)"
        out["tiers"] = {
            "ev": {"status": "unavailable", "reason": reason},
            "stmt": {"status": "unavailable", "reason": reason},
        }
        return out

    # ── Tier-1: per-evidence realized score vs gold ──────────────────────────
    ev_pairs = [(r["our_score"], is_gold_correct(r["gold"]["verdict"]))
                for r in per_ev
                if r.get("gold") and isinstance(r.get("our_score"), (int, float))]
    if ev_pairs:
        ev_scores = [s for s, _ in ev_pairs]
        ev_labels = [y for _, y in ev_pairs]
        # Tier-1 confusion axis: positive = CORRECT (pred_correct = score >= tau).
        ev_block = _metric_block(ev_scores, ev_labels, tau=tau)
        ev_block["confusion"] = {  # override to the correct-positive axis
            k: v for k, v in
            confusion_metrics([(y, (s >= tau)) for s, y in ev_pairs]).items()
            if k in ("tp", "fp", "fn", "tn")
        }
        out["tiers"]["ev"] = {
            "status": "available",
            "n": len(ev_pairs),
            "base_rate_correct": sum(ev_labels) / len(ev_labels),
            "arms": {"score": ev_block},
        }
    else:
        out["tiers"]["ev"] = {"status": "unavailable",
                              "reason": "no per-evidence rows with both gold and a score"}

    # ── Tier-2: per-statement three-way belief vs statement gold ─────────────
    # Build the calibration arc's pair-index from gold_path (NOT the baked per-row
    # gold_map, which is source_hash-keyed) so the join matches calibration_stage0
    # exactly: prefer the (matches_hash, source_hash) pair, fall back to a unique
    # source_hash. Each gold row carries pa_hash (the statement grouping key).
    by_pair: dict[tuple[int, int], dict] = {}
    by_sh_gold: dict[int, list[dict]] = defaultdict(list)
    with open(gold_path) as gf:
        for line in gf:
            line = line.strip()
            if not line:
                continue
            gr = json.loads(line)
            sh = _ukey(gr.get("source_hash"))
            mh = _ukey(gr.get("matches_hash"))
            if sh is None:
                continue
            if mh is not None:
                by_pair[(mh, sh)] = gr
            by_sh_gold[sh].append(gr)

    def _gold_for(er: dict) -> dict | None:
        sh = _ukey(er.get("_source_hash"))
        if sh is None:
            return None
        sx = er.get("_stmt_hash")
        try:
            mh = int(sx, 16) & _HASH_MASK if sx else None
        except (ValueError, TypeError):
            mh = None
        if mh is not None and (mh, sh) in by_pair:
            return by_pair[(mh, sh)]
        cand = by_sh_gold.get(sh, [])
        return cand[0] if len(cand) == 1 else None

    # Group statements by the gold's pa_hash; drop parse-null verdicts (the
    # calibration join_model drops them before grouping).
    by_stmt: dict[Any, dict] = defaultdict(lambda: {"rows": [], "tags": []})
    for a in stmt_agg.values():
        for er in a["belief_rows"]:
            if er.get("verdict") is None:
                continue
            g = _gold_for(er)
            if g is None or not g.get("tag"):
                continue
            rec = by_stmt[g["pa_hash"]]
            rec["rows"].append(er)
            rec["tags"].append(g["tag"])

    stmt_rows: list[dict] = []
    for _pa, rec in by_stmt.items():
        gv = aggregate_gold(rec["tags"])
        if gv is None:
            continue
        # dedup OFF — the calibration arc rolls up raw evidence (no within-source
        # text de-dup), so this matches calibration_ship_gate's belief exactly.
        sb = statement_belief(rec["rows"], RECALIBRATED_PRIORS, dedup=False)
        if sb.belief is None:
            continue  # nothing read → belief UNDEFINED, drop (matches calib scripts)
        soft_b = (statement_belief(rec["rows"], RECALIBRATED_PRIORS, dedup=False, soft=soft).belief
                  if soft else None)
        # Strata metadata for the residual map (I3). stmt_type/bucket_group ride on
        # the belief_rows (build_run_export bakes them). stmt_type can be None (older
        # rows) → that statement is dropped from by_stmt_type; bucket_group is always
        # populated (classify covers every row). driver = which reject path condemned
        # the statement.
        grp = rec["rows"]
        stmt_type = next((r.get("stmt_type") for r in grp if r.get("stmt_type")), None)
        bg = Counter(r.get("bucket_group") for r in grp if r.get("bucket_group"))
        dominant_bucket = bg.most_common(1)[0][0] if bg else None
        driver = ("deterministic" if sb.n_credible_incorrect_det > 0
                  else "llm" if sb.n_credible_incorrect_llm > 0 else "none")
        stmt_rows.append({
            "hard": sb.belief, "parametric": sb.parametric_only, "soft": soft_b,
            "gold_correct": is_gold_correct(gv),
            "verdict_statement": sb.verdict_statement,   # tiered decision (I2)
            "stmt_type": stmt_type,                      # I3 strata dims ↓
            "n_distinct_sources": sb.n_distinct_sources,
            "n_evidence": sb.n_evidence,
            "dominant_bucket": dominant_bucket,
            "driver": driver,
        })

    if stmt_rows:
        labels = [r["gold_correct"] for r in stmt_rows]
        arms: dict = {
            "hard": _metric_block([r["hard"] for r in stmt_rows], labels, tau=tau),
            "parametric": _metric_block([r["parametric"] for r in stmt_rows], labels, tau=tau),
        }
        if soft:
            arms["soft"] = _metric_block([r["soft"] for r in stmt_rows], labels, tau=tau)
        else:
            arms["soft"] = {
                "status": "unavailable",
                "reason": (f"no fitted soft-weight calibration for reader {model!r} "
                           "(only gemma-4-26B / medpsy-4B are fitted)"
                           if model else "no model recorded for this run"),
            }
        out["tiers"]["stmt"] = {
            "status": "available",
            "n": len(stmt_rows),
            "base_rate_correct": sum(labels) / len(labels),
            "arms": arms,
            # ── NEW (schema v2): first-class statement heuristic surface ─────────
            # verdict_err = error-detection F1 on the tiered verdict_statement;
            # stratified = where the residual error mass concentrates (R7 reads it).
            "verdict_err": _verdict_err_confusion(stmt_rows),
            "stratified": {
                "by_stmt_type": _stratify(stmt_rows, lambda r: r["stmt_type"], tau),
                "by_n_sources": _stratify(
                    stmt_rows, lambda r: "multi" if r["n_distinct_sources"] > 1 else "single", tau),
                "by_n_evidence": _stratify(
                    stmt_rows, lambda r: "multi" if r["n_evidence"] > 1 else "single", tau),
                "by_dominant_bucket": _stratify(stmt_rows, lambda r: r["dominant_bucket"], tau),
                "by_driver": _stratify(stmt_rows, lambda r: r["driver"], tau),
            },
        }
    else:
        out["tiers"]["stmt"] = {"status": "unavailable",
                                "reason": "no statements with gold and a defined belief"}
    return out


def build_run_export(
    run_path: str,
    corpus_path: str = DEFAULT_CORPUS,
    *,
    run_id: str | None = None,
    model: str | None = None,
    gold_path: str | None = None,
) -> tuple[list[dict], list[dict], dict, dict]:
    """Enrich a raw run into ``(per_evidence, per_statement, meta, metrics)``.

    Pure transform: dedups the run by ``(stmt_i, evidence_i)`` (last write wins,
    matching the figure reproducers), joins authoritative evidence text + INDRA
    ids from the corpus positionally, classifies buckets, and rolls up per
    statement. ``run_id`` / ``model`` default to the run's ``.meta.json``.

    ``metrics`` is the E5 per-run calibration-product contract (``metrics.json``):
    Tier-1 per-evidence + Tier-2 three-way per-statement ECE/AUROC/AUPRC/Brier +
    reliability bins, keyed per run + model. Named-empty when no gold is baked.
    """
    rmeta = _read_run_meta(run_path)
    run_id = run_id or rmeta.get("run_id")
    # Canonicalize the recorded model name (host-prefix + full tag) so every
    # export — incl. legacy runs recorded under abbreviated names — reads
    # consistently; model_size/_soft_calibration_block downstream use this.
    model = canonical_model_name(model or rmeta.get("model"))

    with open(corpus_path) as f:
        corpus = json.load(f)

    # Per-run gold: baked in at export time from the run's OWN curation source,
    # so the viewer reads gold straight off the run (switches per run, nothing
    # global). None → no gold key written, viewer falls back to its legacy index.
    gold_map = load_gold_map(gold_path) if gold_path else None

    rows: dict[tuple[int, int], dict] = {}
    n_lines = 0
    with open(run_path) as f:
        for line in f:
            if not line.strip():
                continue
            n_lines += 1
            d = json.loads(line)
            rows[(d["stmt_i"], d["evidence_i"])] = d

    per_ev: list[dict] = []
    bucket_n: Counter = Counter()
    join_miss = sourcehash_mismatch = textlen_mismatch = 0
    # run-level observed-cost accumulators (one pass, alongside the row loop)
    cost_total = 0.0
    cost_in_tok = cost_out_tok = 0
    n_rows_costed = n_rows_unavailable = n_rows_no_llm = 0
    run_models: set[str] = set()
    any_unavailable = False
    any_estimated = False
    stmt_agg: dict[str, dict] = defaultdict(lambda: {
        "scores": [], "nc": 0, "ni": 0, "nother": 0, "buckets": Counter(),
        "belief": None, "subject": None, "stmt_type": None, "object": None,
        "stmt_i": None, "matches_hash": None, "indra_id": None,
        "beliefs": set(), "indra_ids": set(), "pmids": set(), "sources": set(),
        # Minimal per-evidence rows for the three-way belief (statement_belief);
        # collected from fields already read off `d`, so no new parsing. Used
        # only to compute the NEW belief_* keys — the existing rollup is untouched.
        "belief_rows": [],
    })

    for (si, ei), d in rows.items():
        ev_parsed, reasoning = split_preview(d.get("raw_text_preview"), d.get("text_len") or 0)
        bucket = classify(d, ev_parsed, reasoning)
        bucket_n[bucket] += 1

        ev_text, matches_hash, indra_id = "", None, None
        if 0 <= si < len(corpus):
            stmt = corpus[si]
            matches_hash = stmt.get("matches_hash")
            indra_id = stmt.get("id")
            evlist = stmt.get("evidence") or []
            if 0 <= ei < len(evlist):
                cev = evlist[ei]
                ev_text = cev.get("text") or ""
                if (cev.get("source_hash") is not None and d.get("source_hash") is not None
                        and cev.get("source_hash") != d.get("source_hash")):
                    sourcehash_mismatch += 1
        tl = d.get("text_len") or 0
        if tl > 0 and not ev_text:
            join_miss += 1
        if tl > 0 and ev_text and abs(len(ev_text) - tl) > 2:
            textlen_mismatch += 1

        obj = d.get("object")
        cl = d.get("call_log") or []
        cost = call_log_cost(cl)
        fin = cl[-1].get("finish_reason") if cl else None
        out_tok = cl[-1].get("out_tokens") if cl else None
        # Last-call reasoning trace (same last-call semantics as fin/out_tok).
        rtrace = compact_reasoning_trace(cl[-1].get("reasoning_trace")) if cl else None
        rchars = len(d.get("raw_text_preview") or "")
        reasoning_truncated = (fin == "length") or bool(out_tok and out_tok * 3.5 > rchars + 200)

        ev_row = {
            "stmt_hash": d.get("stmt_hash"),
            "evidence_hash": d.get("evidence_hash"),
            "source_hash": d.get("source_hash"),
            "indra_matches_hash": matches_hash,
            "indra_id": indra_id,
            "stmt_i": si, "evidence_i": ei,
            "subject": d.get("subject"),
            "stmt_type": d.get("stmt_type"),
            "object": obj if obj not in (None, "") else None,
            "source_api": d.get("source_api"),
            "pmid": d.get("pmid"),
            "evidence_text": ev_text,
            "text_len": tl,
            "rasmachine_belief": _r3(d.get("belief")),
            "our_score": _r3(d.get("score")),
            "verdict": d.get("verdict"),
            "confidence": d.get("confidence"),
            "reasoning": d.get("raw_text_preview"),
            "grounding_status": d.get("grounding_status"),
            "tier": d.get("tier"),
            "provenance_triggered": d.get("provenance_triggered"),
            "error": d.get("error"),
            "latency_s": _r3(d.get("latency_s")),
            "tokens": d.get("tokens"),
            "reasoning_chars": rchars,
            "reasoning_truncated": reasoning_truncated,
            "gen_finish_reason": fin,
            "gen_out_tokens": out_tok,
            # Uniform CoT capture (status + tokens + provenance + committed
            # support/objection). None on legacy rows scored before the trace
            # existed; the viewer falls back to `reasoning` then.
            "reasoning_trace": rtrace,
            # observed LLM cost (computed once here, where the full call_log is in
            # scope). output_tokens is the call_log SUM (distinct from tokens /
            # gen_out_tokens, which are LAST-call only) — do not reconcile them.
            "cost_usd": cost["cost_usd"],            # float (USD) | None
            "cost_status": cost["cost_status"],      # "known" | "unavailable"
            "input_tokens": cost["input_tokens"],    # int — observed prompt tokens, summed over calls
            "output_tokens": cost["output_tokens"],  # int — observed completion tokens, summed over calls
            "n_calls": cost["n_calls"],              # int — LLM calls for this evidence
            "bucket": bucket,
            "bucket_group": GROUP_NAME[META[bucket][0]],
        }
        # Bake gold (GoldVerdict | null) when a gold source is given. Every row
        # gets the key in a baked run (null = uncurated) so the viewer can tell
        # "baked + uncurated" from "legacy run, look in the index". Join on
        # source_hash (the run's authoritative evidence id).
        if gold_map is not None:
            gk = _ukey(d.get("source_hash"))
            ev_row["gold"] = gold_map.get(gk) if gk is not None else None
        per_ev.append(ev_row)

        # accumulate run-level cost (same `cost` dict baked into the row above)
        cost_in_tok += cost["input_tokens"]
        cost_out_tok += cost["output_tokens"]
        run_models.update(cost["models"])
        if cost["cost_status"] == "unavailable":
            any_unavailable = True
            n_rows_unavailable += 1
        elif cost["n_calls"] == 0:
            n_rows_no_llm += 1
        else:
            if cost["cost_status"] == "estimated":
                any_estimated = True
            cost_total += cost["cost_usd"]
            n_rows_costed += 1

        a = stmt_agg[d.get("stmt_hash")]
        v = d.get("verdict")
        if v == "correct":
            a["nc"] += 1
        elif v == "incorrect":
            a["ni"] += 1
        else:
            a["nother"] += 1
        if isinstance(d.get("score"), (int, float)):
            a["scores"].append(d["score"])
        a["buckets"][bucket] += 1
        if a["belief"] is None and isinstance(d.get("belief"), (int, float)):
            a["belief"] = d["belief"]
        if isinstance(d.get("belief"), (int, float)):
            a["beliefs"].add(_r3(d["belief"]))
        if indra_id:
            a["indra_ids"].add(indra_id)
        if a["subject"] is None:
            a["subject"], a["stmt_type"], a["object"] = (
                d.get("subject"), d.get("stmt_type"), obj if obj not in (None, "") else None)
            a["stmt_i"], a["matches_hash"], a["indra_id"] = si, matches_hash, indra_id
        if d.get("pmid"):
            a["pmids"].add(d["pmid"])
        if d.get("source_api"):
            a["sources"].add(d["source_api"])
        # Row dict for statement_belief (hard / parametric / soft). evidence_text
        # is the joined corpus text (for within-source de-dup), evidence_hash a
        # de-dup fallback.
        a["belief_rows"].append({
            "source_api": d.get("source_api"),
            "verdict": d.get("verdict"),
            "confidence": d.get("confidence"),
            "tier": d.get("tier"),
            "evidence_text": ev_text,
            "evidence_hash": d.get("evidence_hash"),
            # Strata metadata for build_run_metrics' Tier-2 residual map (I3);
            # computation-only, never written into per_statement.json.
            "stmt_type": d.get("stmt_type"),
            "bucket_group": GROUP_NAME[META[bucket][0]],
            # gold + statement-grouping keys (used by build_run_metrics' Tier-2
            # join only; never written into per_statement.json). _stmt_hash is the
            # run's statement hash (hex) — the calibration pair-join derives the
            # matches_hash int from it (mirrors calibration_stage0.gold_for).
            "_source_hash": d.get("source_hash"),
            "_stmt_hash": d.get("stmt_hash"),
        })

    # Resolve the run's soft-weight calibration ONCE (frozen per-reader constants;
    # None for an unfitted reader → belief_soft is named-empty per statement).
    calib = calibration_for(model)

    per_stmt: list[dict] = []
    for h, a in stmt_agg.items():
        sc = a["scores"]
        noisy_or = 1 - math.prod(1 - s for s in sc) if sc else None
        dominant = a["buckets"].most_common(1)[0][0] if a["buckets"] else None
        indra_ids = sorted(a["indra_ids"])
        # Three-way belief (NEW; additive — never touches our_noisy_or/mean above).
        # hard = production gated noisy-OR; parametric = ungated; soft = survival-
        # weight recalibration (None when the reader has no fit). verdict_statement
        # is tier-driven — the soft weight moves the scalar, not the decision.
        sb = statement_belief(a["belief_rows"], RECALIBRATED_PRIORS)
        soft_belief = (statement_belief(a["belief_rows"], RECALIBRATED_PRIORS, soft=calib).belief
                       if calib else None)
        # Statement-grain gold (I4): any-incorrect-wins over THIS statement's
        # per-evidence gold tags (gold_map is source_hash-keyed + pre-aggregated
        # per source). None when no evidence is curated. This is the source_hash-
        # keyed, stmt_hash-grouped rollup (load_gold_map's authoritative-source-hash
        # doctrine) — NOT byte-equal to the Tier-2 metrics join, which prefers the
        # (matches_hash, source_hash) pair and groups by the gold's pa_hash. They
        # agree at evidence grain but can diverge for a source_hash shared across
        # statements; this surface is for per-statement viewer drill, not for
        # cross-checking the Tier-2 arms.
        gold_stmt = None
        if gold_map is not None:
            g_tags: list[str] = []
            g_n = 0
            for r in a["belief_rows"]:
                gk = _ukey(r.get("_source_hash"))
                gv = gold_map.get(gk) if gk is not None else None
                if gv:
                    g_tags += gv.get("tags") or []
                    g_n += 1
            gverdict = aggregate_gold(g_tags)
            if gverdict is not None:
                gold_stmt = {"verdict": gverdict, "n": g_n, "tags": sorted(set(g_tags))}
        per_stmt.append({
            "stmt_hash": h,
            "indra_matches_hash": a["matches_hash"],
            "indra_id": a["indra_id"],
            "indra_ids": indra_ids,
            "n_indra_statements": len(indra_ids),
            "stmt_i": a["stmt_i"],
            "subject": a["subject"], "stmt_type": a["stmt_type"], "object": a["object"],
            "rasmachine_belief": _r3(a["belief"]),
            "rasmachine_beliefs": sorted(a["beliefs"]),
            "n_evidence": a["nc"] + a["ni"] + a["nother"],
            "n_correct": a["nc"], "n_incorrect": a["ni"], "n_unscored": a["nother"],
            "our_mean_score": _r3(sum(sc) / len(sc)) if sc else None,
            "our_noisy_or": _r3(noisy_or),
            "our_max_score": _r3(max(sc)) if sc else None,
            "our_min_score": _r3(min(sc)) if sc else None,
            "dominant_bucket": dominant,
            "bucket_counts": dict(a["buckets"]),
            "pmids": sorted(a["pmids"]),
            "sources": sorted(a["sources"]),
            # ── E5: three-way calibrated belief (NEW; purely additive) ─────────
            "belief_hard": _r3(sb.belief),               # gated noisy-OR (production hard gate)
            "belief_parametric": _r3(sb.parametric_only),  # no gating — all surviving evidence
            "belief_soft": _r3(soft_belief),             # soft survival weight; None = no fit
            "belief_verdict_statement": sb.verdict_statement,  # correct|review|incorrect (tier-driven)
            # ── I4 + E10 (schema v7; purely additive) ──────────────────────────
            # gold_statement: aggregated statement-grain gold {verdict,n,tags} | null.
            # coherence_summary: the multi-evidence depth behind the belief (post-
            # dedup; the headline n_correct/n_incorrect above stay RAW). These make
            # the belief_* signal a first-class, joinable statement surface.
            "gold_statement": gold_stmt,
            "coherence_summary": {
                # ALL counts here are POST-dedup (the belief's own view); the
                # headline n_correct/n_incorrect above are RAW pre-dedup. n_correct/
                # n_incorrect are included so the post-dedup correct/incorrect split
                # the belief actually rests on is reconstructable, not just its
                # credible-incorrect subset.
                "n_dedup_groups": sb.n_dedup_groups,
                "n_distinct_sources": sb.n_distinct_sources,
                "n_correct": sb.n_correct,
                "n_incorrect": sb.n_incorrect,
                "n_no_text": sb.n_no_text,
                "n_parse_fail": sb.n_parse_fail,
                "n_null_source": sb.n_null_source,
                "n_credible_incorrect_det": sb.n_credible_incorrect_det,
                "n_credible_incorrect_llm": sb.n_credible_incorrect_llm,
            },
        })

    # internal consistency (the partition is total; tallies match depth)
    assert len(rows) == sum(bucket_n.values()) == len(per_ev), "row-count drift"
    for s in per_stmt:
        assert s["n_correct"] + s["n_incorrect"] + s["n_unscored"] == s["n_evidence"], (
            f"verdict tally != depth: {s['stmt_hash']}")

    n_collide = sum(1 for s in per_stmt if s["n_indra_statements"] > 1)
    n_belief_collide = sum(1 for s in per_stmt if len(s["rasmachine_beliefs"]) > 1)
    meta = {
        "title": "RasMachine belief vs. LLM per-evidence belief signal — raw results",
        "generated_date": datetime.date.today().isoformat(),
        "run_id": run_id,
        "model": model,
        "generated_from": {"run": run_path, "corpus": corpus_path, "gold": gold_path},
        "companion_report": "reports/rasmachine_belief_comparison.html",
        "gold": ({
            "source": gold_path,
            "covered": sum(1 for r in per_ev if r.get("gold")),
            "total": len(per_ev),
        } if gold_map is not None else None),
        "counts": {
            "run_lines": n_lines, "unique_evidence_rows": len(per_ev),
            "statements": len(per_stmt),
            "statements_scored": sum(1 for s in per_stmt if s["n_correct"] + s["n_incorrect"] > 0),
            "statements_collapsing_multiple_indra": n_collide,
            "statements_with_divergent_belief": n_belief_collide,
        },
        "bucket_counts": {b: bucket_n[b] for b in ORDER},
        "join_quality": {
            "evidence_text_misses": join_miss,
            "source_hash_mismatches": sourcehash_mismatch,
            "text_len_mismatches": textlen_mismatch,
        },
        "reasoning_quality": {
            "rows_with_reasoning": sum(1 for r in per_ev if r["reasoning"]),
            "reasoning_truncated": sum(1 for r in per_ev if r["reasoning_truncated"]),
            "generation_length_capped": sum(1 for r in per_ev if r["gen_finish_reason"] == "length"),
            # CoT-access histogram: how many evidences had readable CoT vs sealed
            # (encrypted) vs withheld vs none — the epistemic-access distribution
            # across whatever models scored the run. None = legacy row, no trace.
            "trace_status": _count_trace_status(per_ev),
            "note": "reasoning is a clipped ~1000-char preview for the truncated rows; the full "
                    "chain-of-thought was not recorded. verdict/confidence/score were parsed before "
                    "clipping and are unaffected. reasoning_trace carries the per-evidence CoT-access "
                    "status (plaintext/inline/encrypted/not_returned/none) + committed support/objection.",
        },
        "cost": {
            # "known": every priced row used an observed list price (rows may be $0).
            # "estimated": all rows priced, but some used a Bedrock-grounded estimate
            #              (a local/self-hosted model with no observed list price).
            # "partial": some rows priced, some unavailable -> total covers only priced rows.
            # "unavailable": NO row had a verified price (whole run is unverified).
            "status": ("unavailable" if n_rows_costed == 0 and any_unavailable
                       else "partial" if any_unavailable
                       else "estimated" if any_estimated
                       else "known"),
            "total_usd": round(cost_total, 4) if n_rows_costed > 0 else None,
            "input_tokens": cost_in_tok,
            "output_tokens": cost_out_tok,
            "n_evidence_costed": n_rows_costed,            # rows that contributed to total_usd (>=1 priced call)
            "n_evidence_no_llm": n_rows_no_llm,            # rows with 0 LLM calls ($0, no spend)
            "n_evidence_unavailable": n_rows_unavailable,  # rows withheld from total (unverified/missing model)
            "models": sorted(run_models),                  # distinct model_ids observed across the run
            # USD per 1k scored evidences — comparable across runs of different size.
            "usd_per_1k_evidence": (round(cost_total / n_rows_costed * 1000, 4)
                                    if n_rows_costed > 0 else None),
        },
        # 'soft_calibration' (not 'calibration') to avoid collision with the
        # viewer's existing Validity.calibration (belief-vs-INDRA residual {n,mae,bias}).
        "soft_calibration": _soft_calibration_block(model),
        # Ground-truth model size (params), so F1 can be read over scale, not just
        # cost. Static model metadata baked per-run (travels with the run; viewer
        # holds no size table). Closed models -> status 'unknown' (never guessed).
        "model_meta": model_size(model or "unknown"),
        # v6: + metrics.json calibration products (Tier-1/Tier-2) + per-statement
        # three-way belief (belief_hard/parametric/soft + belief_verdict_statement)
        # v7: + per-statement gold_statement (aggregated statement-grain gold) +
        # coherence_summary (multi-evidence depth behind the belief)
        "schema_version": 7,
    }
    metrics = build_run_metrics(per_ev, stmt_agg, gold_map, model, run_id, gold_path)
    return per_ev, per_stmt, meta, metrics


def write_run_export(
    run_path: str,
    corpus_path: str = DEFAULT_CORPUS,
    out_dir: str | None = None,
    *,
    run_id: str | None = None,
    model: str | None = None,
    gold_path: str | None = None,
) -> dict:
    """Build + write the export for ``run_path``. Returns the meta dict.

    ``out_dir`` defaults to ``data/exports/<run_id>/`` so each run gets its own
    self-describing export the viewer discovers by globbing ``export_meta.json``.
    ``gold_path`` (optional) bakes per-run gold from that curation source.
    """
    per_ev, per_stmt, meta, metrics = build_run_export(
        run_path, corpus_path, run_id=run_id, model=model, gold_path=gold_path)
    out_dir = out_dir or os.path.join("data", "exports", str(meta["run_id"]))
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "per_evidence.jsonl"), "w") as f:
        for rec in per_ev:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "per_statement.json"), "w") as f:
        json.dump(per_stmt, f, ensure_ascii=False)
    with open(os.path.join(out_dir, "export_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    # E5: the per-run calibration-product contract the viewer (C4/C5) serves
    # byte-exact (no downstream recompute).
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return meta


def _main() -> int:
    ap = argparse.ArgumentParser(description="Enrich a raw monolithic run into the viewer/collaborator export.")
    ap.add_argument("run", help="path to data/results/<run>.jsonl")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--out", default=None, help="export dir (default: data/exports/<run_id>/)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--gold", default=None,
                    help="curation JSONL to bake per-run gold from (pa_hash/source_hash/tag)")
    args = ap.parse_args()
    meta = write_run_export(args.run, args.corpus, args.out,
                            run_id=args.run_id, model=args.model, gold_path=args.gold)
    out = args.out or os.path.join("data", "exports", str(meta["run_id"]))
    c = meta["counts"]
    print(f"wrote {out}/  ·  {c['unique_evidence_rows']} evidence · {c['statements']} statements")
    if meta.get("gold"):
        print(f"  gold baked: {meta['gold']['covered']}/{meta['gold']['total']} evidences curated")
    for b in ORDER:
        print(f"  {b:<22} {meta['bucket_counts'][b]:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
