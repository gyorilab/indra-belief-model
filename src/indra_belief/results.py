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
lives here so the export and the report's figure scripts share one definition
(``scripts/fig1_drilldown_data`` re-exports these for back-compat).

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

from indra_belief.curation import aggregate_gold
from indra_belief.corpus.cost import price_basis, token_cost_usd
from indra_belief.model_client import canonical_model_name
from indra_belief.model_meta import model_size
from indra_belief.calibration_constants import calibration_for

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


def build_run_export(
    run_path: str,
    corpus_path: str = DEFAULT_CORPUS,
    *,
    run_id: str | None = None,
    model: str | None = None,
    gold_path: str | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Enrich a raw run into ``(per_evidence, per_statement, meta)``.

    Pure transform: dedups the run by ``(stmt_i, evidence_i)`` (last write wins,
    matching the figure reproducers), joins authoritative evidence text + INDRA
    ids from the corpus positionally, classifies buckets, and rolls up per
    statement. ``run_id`` / ``model`` default to the run's ``.meta.json``.
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

    per_stmt: list[dict] = []
    for h, a in stmt_agg.items():
        sc = a["scores"]
        noisy_or = 1 - math.prod(1 - s for s in sc) if sc else None
        dominant = a["buckets"].most_common(1)[0][0] if a["buckets"] else None
        indra_ids = sorted(a["indra_ids"])
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
        "schema_version": 5,  # v5: per-evidence reasoning_trace (CoT-access + committed justification)
    }
    return per_ev, per_stmt, meta


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
    per_ev, per_stmt, meta = build_run_export(
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
