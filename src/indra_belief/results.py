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
        if text_len and len(cand) <= max(text_len + 30, int(text_len * 1.3)):
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
    if tl < 30:
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


def build_run_export(
    run_path: str,
    corpus_path: str = DEFAULT_CORPUS,
    *,
    run_id: str | None = None,
    model: str | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Enrich a raw run into ``(per_evidence, per_statement, meta)``.

    Pure transform: dedups the run by ``(stmt_i, evidence_i)`` (last write wins,
    matching the figure reproducers), joins authoritative evidence text + INDRA
    ids from the corpus positionally, classifies buckets, and rolls up per
    statement. ``run_id`` / ``model`` default to the run's ``.meta.json``.
    """
    rmeta = _read_run_meta(run_path)
    run_id = run_id or rmeta.get("run_id")
    model = model or rmeta.get("model")

    with open(corpus_path) as f:
        corpus = json.load(f)

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
        fin = cl[-1].get("finish_reason") if cl else None
        out_tok = cl[-1].get("out_tokens") if cl else None
        rchars = len(d.get("raw_text_preview") or "")
        reasoning_truncated = (fin == "length") or bool(out_tok and out_tok * 3.5 > rchars + 200)

        per_ev.append({
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
            "bucket": bucket,
            "bucket_group": GROUP_NAME[META[bucket][0]],
        })

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
        "generated_from": {"run": run_path, "corpus": corpus_path},
        "companion_report": "reports/rasmachine_belief_comparison.html",
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
            "note": "reasoning is a clipped ~1000-char preview for the truncated rows; the full "
                    "chain-of-thought was not recorded. verdict/confidence/score were parsed before "
                    "clipping and are unaffected.",
        },
        "schema_version": 2,
    }
    return per_ev, per_stmt, meta


def write_run_export(
    run_path: str,
    corpus_path: str = DEFAULT_CORPUS,
    out_dir: str | None = None,
    *,
    run_id: str | None = None,
    model: str | None = None,
) -> dict:
    """Build + write the export for ``run_path``. Returns the meta dict.

    ``out_dir`` defaults to ``data/exports/<run_id>/`` so each run gets its own
    self-describing export the viewer discovers by globbing ``export_meta.json``.
    """
    per_ev, per_stmt, meta = build_run_export(run_path, corpus_path, run_id=run_id, model=model)
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
    args = ap.parse_args()
    meta = write_run_export(args.run, args.corpus, args.out, run_id=args.run_id, model=args.model)
    out = args.out or os.path.join("data", "exports", str(meta["run_id"]))
    c = meta["counts"]
    print(f"wrote {out}/  ·  {c['unique_evidence_rows']} evidence · {c['statements']} statements")
    for b in ORDER:
        print(f"  {b:<22} {meta['bucket_counts'][b]:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
