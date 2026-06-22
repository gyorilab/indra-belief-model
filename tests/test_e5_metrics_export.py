"""E5 — per-run calibration-product export (the math→viewer seam).

Locks the three E5 guarantees:

1. GOLDEN identity — the pre-existing per_evidence/per_statement output is
   unchanged: per_evidence.jsonl is untouched (no soft belief per evidence), and
   per_statement.json grows by EXACTLY the four additive ``belief_*`` keys (every
   pre-existing key/value bit-for-bit identical). Proven on a synthetic run here;
   the data-backed byte-identity (eval_curation_v1) is checked in
   ``test_e5_crosscheck_*`` when the corpus/run are present.

2. metrics.json CONTRACT — the C4/C5 schema: schema_version, two tiers
   (ev/stmt), the stable per-arm block {n, ece, auroc, auprc, brier,
   reliability, resolution, uncertainty, confusion{tp,fp,fn,tn}, bins[8]},
   named-empty tiers/arms (status+reason, no imputed zeros).

3. CROSS-CHECK — metrics.json tiers.stmt.arms.{hard,parametric,soft}.{ece,auroc}
   equal calibration_ship_gate.json's metrics.{hard,parametric,guard} for the
   same run. hard/parametric are byte-exact (no weights); soft matches within the
   3-dp frozen-constant rounding (documented in calibration_constants).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from indra_belief.results import build_run_export, build_run_metrics, load_gold_map

ROOT = Path(__file__).resolve().parents[1]
NEW_KEYS = {"belief_hard", "belief_parametric", "belief_soft", "belief_verdict_statement"}
PER_ARM_KEYS = {"n", "ece", "auroc", "auprc", "brier", "reliability",
                "resolution", "uncertainty", "confusion", "bins"}


# ── fixtures (synthetic; no large data) ──────────────────────────────────────

def _corpus():
    return [{
        "matches_hash": "123", "id": "id1", "belief": 0.9,
        "evidence": [
            {"text": "MEK phosphorylates ERK in cells normally.", "source_hash": 11},
            {"text": "MEK does not bind DNA at all.", "source_hash": 22},
            {"text": "RAF activates MEK strongly here.", "source_hash": 33},
        ],
    }]


def _row(ei, sh, verdict, score, conf="high"):
    return {
        "stmt_i": 0, "evidence_i": ei, "stmt_hash": "7b", "evidence_hash": f"e{ei}",
        "source_hash": sh, "subject": "MAP2K1", "stmt_type": "Phosphorylation",
        "object": "MAPK1", "source_api": "reach", "pmid": "1", "text_len": 40,
        "belief": 0.9, "score": score, "verdict": verdict, "confidence": conf,
        "raw_text_preview": "[TIER 2 LLM]\nYes.", "grounding_status": "all_match",
        "tier": "llm_comprehension", "provenance_triggered": False, "error": None,
        "latency_s": 1.0, "tokens": 10, "call_log": [],
    }


def _write(tmp_path, rows, corpus):
    run = tmp_path / "run.jsonl"
    run.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    corp = tmp_path / "corpus.json"
    corp.write_text(json.dumps(corpus))
    return str(run), str(corp)


def _gold(tmp_path, gold_rows):
    g = tmp_path / "gold.jsonl"
    g.write_text("\n".join(json.dumps(r) for r in gold_rows) + "\n")
    return str(g)


# ── 1. GOLDEN identity: per_statement grows by exactly the 4 belief_* keys ────

def test_per_statement_is_additive_only(tmp_path):
    rows = [_row(0, 11, "correct", 0.95), _row(1, 22, "incorrect", 0.05),
            _row(2, 33, "correct", 0.8)]
    run, corp = _write(tmp_path, rows, _corpus())

    # Run WITHOUT the belief_* keys would be the pre-E5 shape; we assert here that
    # the only keys beyond a fixed pre-E5 set are the four belief_* keys.
    _ev, per_stmt, _meta, _metrics = build_run_export(run, corp, run_id="r", model="gemma")
    assert len(per_stmt) == 1
    s = per_stmt[0]
    assert NEW_KEYS <= set(s)
    # the four are the ONLY belief_* keys; the rollup keys are still present + scalar
    assert s["our_noisy_or"] is not None and s["our_mean_score"] is not None
    assert s["belief_hard"] is not None
    assert s["belief_parametric"] is not None
    # gemma has a fit → soft present
    assert s["belief_soft"] is not None
    assert s["belief_verdict_statement"] in ("correct", "review", "incorrect")


def test_per_statement_no_internal_join_keys_leak(tmp_path):
    # belief_rows' join helpers (_source_hash/_stmt_hash) must NOT appear in the
    # written per_statement.json — they are computation-only.
    rows = [_row(0, 11, "correct", 0.95)]
    run, corp = _write(tmp_path, rows, _corpus())
    _ev, per_stmt, _meta, _metrics = build_run_export(run, corp, run_id="r", model="gemma")
    leaked = {k for s in per_stmt for k in s if k.startswith("_") or k == "belief_rows"}
    assert leaked == set(), leaked


def test_per_evidence_has_no_soft_belief(tmp_path):
    # The soft weight is a statement-level recalibration; there is no per-evidence
    # soft belief. per_evidence rows must carry no belief_* / soft key (keeps the
    # file byte-identical to pre-E5).
    rows = [_row(0, 11, "correct", 0.95)]
    run, corp = _write(tmp_path, rows, _corpus())
    per_ev, _ps, _meta, _metrics = build_run_export(run, corp, run_id="r", model="gemma")
    assert not any(k.startswith("belief_") or k == "belief_soft" for k in per_ev[0])


def test_no_soft_belief_when_reader_unfitted(tmp_path):
    # An unfitted reader (e.g. an unknown model) → belief_soft is None (named-empty),
    # never an imputed 0.0; hard/parametric still computed.
    rows = [_row(0, 11, "correct", 0.95)]
    run, corp = _write(tmp_path, rows, _corpus())
    _ev, per_stmt, _meta, _metrics = build_run_export(run, corp, run_id="r", model="mystery-7b")
    s = per_stmt[0]
    assert s["belief_soft"] is None
    assert s["belief_hard"] is not None and s["belief_parametric"] is not None


# ── 2. metrics.json contract (schema shape) ──────────────────────────────────

def _assert_arm_shape(arm):
    assert set(arm) == PER_ARM_KEYS, set(arm) ^ PER_ARM_KEYS
    assert len(arm["bins"]) == 8  # always the BINS_8 partition
    assert set(arm["confusion"]) == {"tp", "fp", "fn", "tn"}
    for b in arm["bins"]:
        assert set(b) == {"lo", "hi", "n", "mean_pred", "empirical"}
        if b["n"] == 0:
            assert b["mean_pred"] is None and b["empirical"] is None


def test_metrics_schema_with_gold(tmp_path):
    rows = [_row(0, 11, "correct", 0.95), _row(1, 22, "incorrect", 0.05),
            _row(2, 33, "correct", 0.8)]
    run, corp = _write(tmp_path, rows, _corpus())
    gold = _gold(tmp_path, [
        {"matches_hash": 123, "source_hash": 11, "pa_hash": 900, "tag": "correct"},
        {"matches_hash": 123, "source_hash": 22, "pa_hash": 900, "tag": "wrong_relation"},
        {"matches_hash": 123, "source_hash": 33, "pa_hash": 900, "tag": "correct"},
    ])
    _ev, _ps, _meta, m = build_run_export(run, corp, run_id="r", model="gemma", gold_path=gold)

    assert m["schema_version"] == 2
    assert m["run_id"] == "r" and m["model"] == "gemma"
    assert set(m["tiers"]) == {"ev", "stmt"}
    assert m["metrics_basis"]["bins"] == "BINS_8"
    assert m["metrics_basis"]["tau"] == 0.5
    assert m["metrics_basis"]["soft_calibration"]["status"] == "available"

    ev = m["tiers"]["ev"]
    assert ev["status"] == "available" and ev["n"] == 3
    _assert_arm_shape(ev["arms"]["score"])

    st = m["tiers"]["stmt"]
    assert st["status"] == "available"
    assert set(st["arms"]) == {"hard", "parametric", "soft"}
    for arm in st["arms"].values():
        _assert_arm_shape(arm)


def test_metrics_named_empty_without_gold(tmp_path):
    rows = [_row(0, 11, "correct", 0.95)]
    run, corp = _write(tmp_path, rows, _corpus())
    _ev, _ps, _meta, m = build_run_export(run, corp, run_id="r", model="gemma")  # no gold
    assert m["gold"] is None
    for tier in ("ev", "stmt"):
        assert m["tiers"][tier]["status"] == "unavailable"
        assert m["tiers"][tier]["reason"]
        assert "arms" not in m["tiers"][tier]


def test_metrics_soft_arm_named_empty_for_unfitted_reader(tmp_path):
    rows = [_row(0, 11, "correct", 0.95), _row(1, 22, "incorrect", 0.05)]
    run, corp = _write(tmp_path, rows, _corpus())
    gold = _gold(tmp_path, [
        {"matches_hash": 123, "source_hash": 11, "pa_hash": 900, "tag": "correct"},
        {"matches_hash": 123, "source_hash": 22, "pa_hash": 900, "tag": "wrong_relation"},
    ])
    _ev, _ps, _meta, m = build_run_export(run, corp, run_id="r", model="mystery-7b", gold_path=gold)
    st = m["tiers"]["stmt"]
    # hard + parametric still render (two-of-three); soft is named-empty
    _assert_arm_shape(st["arms"]["hard"])
    _assert_arm_shape(st["arms"]["parametric"])
    assert st["arms"]["soft"]["status"] == "unavailable"
    assert st["arms"]["soft"]["reason"]
    assert "ece" not in st["arms"]["soft"]


def test_metrics_written_to_disk_byte_exact(tmp_path):
    # The served numbers must equal metrics.json byte-exact (no downstream
    # recompute) — lock that write_run_export persists exactly the dict.
    from indra_belief.results import write_run_export
    rows = [_row(0, 11, "correct", 0.95), _row(1, 22, "incorrect", 0.05)]
    run, corp = _write(tmp_path, rows, _corpus())
    gold = _gold(tmp_path, [
        {"matches_hash": 123, "source_hash": 11, "pa_hash": 900, "tag": "correct"},
        {"matches_hash": 123, "source_hash": 22, "pa_hash": 900, "tag": "wrong_relation"},
    ])
    out = tmp_path / "export"
    write_run_export(run, corp, str(out), run_id="r", model="gemma", gold_path=gold)
    assert (out / "metrics.json").exists()
    disk = json.loads((out / "metrics.json").read_text())
    _, _, _, mem = build_run_export(run, corp, run_id="r", model="gemma", gold_path=gold)
    assert disk == mem


# ── 3. cross-check vs calibration_ship_gate.py (data-gated) ───────────────────

_SHIP_GATE = ROOT / "data" / "results" / "calibration_ship_gate.json"
_HOLDOUT_GOLD = ROOT / "data" / "benchmark" / "holdout_cc.jsonl"
_RUNS = {
    "gemma": (ROOT / "data" / "results" / "holdout_cc_gemma.jsonl", "gemma-26B"),
    "medpsy": (ROOT / "data" / "results" / "holdout_cc_medpsy.jsonl", "MedPsy-4B"),
}


def _synth_stmt_agg(run_path: Path) -> dict:
    """The minimal stmt_agg build_run_metrics needs, from a raw run — mirrors the
    belief_rows build_run_export collects (no corpus needed; Tier-2 joins on
    _stmt_hash + _source_hash)."""
    rows = {}
    for line in open(run_path):
        if line.strip():
            d = json.loads(line)
            rows[(d["stmt_i"], d["evidence_i"])] = d
    agg = defaultdict(lambda: {"belief_rows": []})
    for d in rows.values():
        agg[d.get("stmt_hash")]["belief_rows"].append({
            "source_api": d.get("source_api"), "verdict": d.get("verdict"),
            "confidence": d.get("confidence"), "tier": d.get("tier"),
            "evidence_text": "", "evidence_hash": d.get("evidence_hash"),
            "_source_hash": d.get("source_hash"), "_stmt_hash": d.get("stmt_hash"),
        })
    return agg


def _synth_per_ev(run_path: Path, gold_map: dict) -> list[dict]:
    mask = (1 << 64) - 1
    rows = {}
    for line in open(run_path):
        if line.strip():
            d = json.loads(line)
            rows[(d["stmt_i"], d["evidence_i"])] = d
    out = []
    for d in rows.values():
        try:
            gk = int(d.get("source_hash")) & mask
        except (TypeError, ValueError):
            gk = None
        s = d.get("score")
        out.append({"our_score": round(s, 3) if isinstance(s, (int, float)) else None,
                    "gold": gold_map.get(gk) if gk is not None else None})
    return out


@pytest.mark.parametrize("model", ["gemma", "medpsy"])
def test_e5_crosscheck_ship_gate(model):
    run_path, sg_name = _RUNS[model]
    if not (_SHIP_GATE.exists() and _HOLDOUT_GOLD.exists() and run_path.exists()):
        pytest.skip("holdout_cc run / gold / ship-gate artifact not present")

    gold_map = load_gold_map(str(_HOLDOUT_GOLD))
    agg = _synth_stmt_agg(run_path)
    per_ev = _synth_per_ev(run_path, gold_map)
    m = build_run_metrics(per_ev, agg, gold_map, model, f"holdout_cc_{model}", str(_HOLDOUT_GOLD))

    ship = {r["name"]: r for r in json.loads(_SHIP_GATE.read_text())}[sg_name]
    sg = ship["eval"]["metrics"]
    stmt = m["tiers"]["stmt"]

    # same statement set (pair-join + pa_hash grouping + dedup off)
    assert stmt["n"] == ship["eval"]["n_test"]

    # hard + parametric: byte-exact (no weights → identical math through src/)
    for arm, sgkey in [("hard", "hard"), ("parametric", "parametric")]:
        a = stmt["arms"][arm]
        assert a["ece"] == pytest.approx(sg[sgkey]["ece"], abs=1e-12), arm
        assert a["auroc"] == pytest.approx(sg[sgkey]["auroc"], abs=1e-12), arm

    # soft: matches within the frozen-constant 3-dp rounding (documented in
    # calibration_constants — the ship gate re-fits the SAME values unrounded).
    a = stmt["arms"]["soft"]
    assert a["ece"] == pytest.approx(sg["guard"]["ece"], abs=2e-4)
    assert a["auroc"] == pytest.approx(sg["guard"]["auroc"], abs=2e-3)


# ── 4. statement-heuristics instrument (I1–I4, E10) — additive + first-class ───

V7_KEYS = {"gold_statement", "coherence_summary"}
COHERENCE_KEYS = {"n_dedup_groups", "n_distinct_sources", "n_correct", "n_incorrect",
                  "n_no_text", "n_parse_fail", "n_null_source",
                  "n_credible_incorrect_det", "n_credible_incorrect_llm"}
VERDICT_ERR_KEYS = {"n", "tp", "fp", "fn", "tn", "accuracy", "precision", "recall", "f1"}
STRATA_DIMS = {"by_stmt_type", "by_n_sources", "by_n_evidence", "by_dominant_bucket", "by_driver"}


def _three_ev_gold(tmp_path):
    """One statement, three reads (2 correct + 1 high-conf incorrect → gold
    incorrect by any-incorrect-wins; verdict_statement → 'review')."""
    rows = [_row(0, 11, "correct", 0.95), _row(1, 22, "incorrect", 0.05),
            _row(2, 33, "correct", 0.8)]
    run, corp = _write(tmp_path, rows, _corpus())
    gold = _gold(tmp_path, [
        {"matches_hash": 123, "source_hash": 11, "pa_hash": 900, "tag": "correct"},
        {"matches_hash": 123, "source_hash": 22, "pa_hash": 900, "tag": "wrong_relation"},
        {"matches_hash": 123, "source_hash": 33, "pa_hash": 900, "tag": "correct"},
    ])
    return run, corp, gold


def test_per_statement_v7_gold_and_coherence_additive(tmp_path):
    # per_statement grows by EXACTLY gold_statement + coherence_summary on top of
    # the v6 belief_* keys; every pre-existing key/value untouched.
    run, corp, gold = _three_ev_gold(tmp_path)
    _ev, per_stmt, meta, _m = build_run_export(run, corp, run_id="r", model="gemma", gold_path=gold)
    s = per_stmt[0]
    assert meta["schema_version"] == 7
    assert NEW_KEYS <= set(s) and V7_KEYS <= set(s)            # v6 + v7 both present
    # gold_statement: any-incorrect-wins over the statement's evidence gold
    assert s["gold_statement"]["verdict"] == "incorrect"
    assert s["gold_statement"]["n"] == 3
    assert "wrong_relation" in s["gold_statement"]["tags"]
    # coherence_summary: the multi-evidence depth (3 reads, all one source)
    assert set(s["coherence_summary"]) == COHERENCE_KEYS
    assert s["coherence_summary"]["n_distinct_sources"] == 1
    # the join helpers still never leak
    assert not any(k.startswith("_") or k == "belief_rows" for k in s)


def test_gold_statement_null_without_gold(tmp_path):
    # No gold baked → gold_statement is null (named-empty), never imputed.
    rows = [_row(0, 11, "correct", 0.95)]
    run, corp = _write(tmp_path, rows, _corpus())
    _ev, per_stmt, _meta, _m = build_run_export(run, corp, run_id="r", model="gemma")
    assert per_stmt[0]["gold_statement"] is None
    # coherence_summary is gold-independent — always present
    assert set(per_stmt[0]["coherence_summary"]) == COHERENCE_KEYS


def test_metrics_statement_verdict_err_review_is_positive(tmp_path):
    # tiers.stmt gains verdict_err (error-detection on the TIERED verdict) as a
    # sibling of arms; arms stay byte-identical (the cross-check proves the math).
    run, corp, gold = _three_ev_gold(tmp_path)
    _ev, _ps, _meta, m = build_run_export(run, corp, run_id="r", model="gemma", gold_path=gold)
    st = m["tiers"]["stmt"]
    assert set(st["arms"]) == {"hard", "parametric", "soft"}      # unchanged
    assert set(st["verdict_err"]) == VERDICT_ERR_KEYS
    # gold=incorrect, verdict=review → review counts as a FLAG (positive=error) → tp
    assert st["verdict_err"]["n"] == 1
    assert st["verdict_err"]["tp"] == 1 and st["verdict_err"]["fn"] == 0
    assert st["verdict_err"]["f1"] == 1.0


def test_metrics_statement_stratified_shape(tmp_path):
    run, corp, gold = _three_ev_gold(tmp_path)
    _ev, _ps, _meta, m = build_run_export(run, corp, run_id="r", model="gemma", gold_path=gold)
    strat = m["tiers"]["stmt"]["stratified"]
    assert set(strat) == STRATA_DIMS
    # one Phosphorylation statement, single source, multi-evidence, llm-driven reject
    assert strat["by_stmt_type"]["Phosphorylation"]["n"] == 1
    assert "single" in strat["by_n_sources"] and "multi" in strat["by_n_evidence"]
    assert "llm" in strat["by_driver"]
    block = strat["by_stmt_type"]["Phosphorylation"]
    assert set(block) == {"n", "base_rate_correct", "verdict_err", "hard"}
    _assert_arm_shape(block["hard"])                              # reuses the arm unit
    assert set(block["verdict_err"]) == VERDICT_ERR_KEYS


def test_metrics_schema_version_is_2(tmp_path):
    run, corp, gold = _three_ev_gold(tmp_path)
    _ev, _ps, _meta, m = build_run_export(run, corp, run_id="r", model="gemma", gold_path=gold)
    assert m["schema_version"] == 2


def test_metrics_gold_without_pa_hash_degrades(tmp_path):
    # rasmachine-style gold (matches_hash/source_hash/tag, NO pa_hash) must NOT
    # crash build_run_metrics — Tier-2 statement falls through to named-empty while
    # Tier-1 (ev) still renders and per_statement still carries gold_statement
    # (which joins on source_hash, not pa_hash).
    rows = [_row(0, 11, "correct", 0.95), _row(1, 22, "incorrect", 0.05)]
    run, corp = _write(tmp_path, rows, _corpus())
    gold = _gold(tmp_path, [
        {"matches_hash": 123, "source_hash": 11, "tag": "correct"},
        {"matches_hash": 123, "source_hash": 22, "tag": "wrong_relation"},
    ])
    _ev, per_stmt, _meta, m = build_run_export(run, corp, run_id="r", model="gemma", gold_path=gold)
    assert m["tiers"]["ev"]["status"] == "available"        # Tier-1 unaffected
    assert m["tiers"]["stmt"]["status"] == "unavailable"    # no pa_hash → named-empty
    assert m["tiers"]["stmt"]["reason"]
    # per_statement gold_statement still resolves (source_hash join)
    assert per_stmt[0]["gold_statement"] is not None
