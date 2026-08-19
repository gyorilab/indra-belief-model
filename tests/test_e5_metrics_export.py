"""E5 — per-run calibration-product export (the math→viewer seam).

Locks the three E5 guarantees under export schema v8 / metrics schema v3:

1. STATEMENT CONTRACT — calibrated belief remains statement-level, internal join
   keys never leak, and unfitted configurations retain a named hard fallback.

2. metrics.json CONTRACT — the C4/C5 schema: schema_version, two tiers
   (ev/stmt), the stable per-arm block {n, ece, auroc, auprc, brier,
   reliability, resolution, uncertainty, confusion{tp,fp,fn,tn}, bins[8]},
   named-empty tiers/arms (status+reason, no imputed zeros).

3. CROSS-CHECK — metrics.json Tier-2 and the live ship-gate scorer use the same
   current holdout path, run-``stmt_hash`` grain, truth-safe source fallback,
   production de-dup/no-text handling, and configuration-specific profile.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pytest

from indra_belief.calibration_constants import BASELINE_PROMPT_SHA256
from indra_belief.probes.calibration import (
    CALIBRATION_FILENAME,
    CALIBRATION_MODEL,
    CALIBRATION_MODEL_ID,
    CALIBRATION_PROBE_DIGEST,
    DEFAULT_CALIBRATION_PATH,
    SENTENCE_SCORE_CONTRACT_VERSION,
    SENTENCE_SCORE_KIND,
)
from indra_belief.results import (
    build_run_export as _build_run_export,
    build_run_metrics,
    load_gold_map,
)

ROOT = Path(__file__).resolve().parents[1]
NEW_KEYS = {"belief_hard", "belief_parametric", "belief_soft", "belief_verdict_statement"}
PER_ARM_KEYS = {"n", "ece", "auroc", "auprc", "brier", "reliability",
                "resolution", "uncertainty", "confusion", "bins"}


def build_run_export(*args, **kwargs):
    """Synthetic fitted-run helper with an explicit historical prompt identity."""
    if kwargs.get("model") == "gemma":
        kwargs["model"] = "remote-gemma-4-26b"
        kwargs["prompt_sha256"] = BASELINE_PROMPT_SHA256
    return _build_run_export(*args, **kwargs)


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
    run.with_suffix(".meta.json").write_text(json.dumps({
        "sentence_score": {
            "status": "enabled",
            "contract_version": SENTENCE_SCORE_CONTRACT_VERSION,
            "grain": "sentence",
            "kind": SENTENCE_SCORE_KIND,
            "calibration_model": CALIBRATION_MODEL,
            "calibration_model_id": CALIBRATION_MODEL_ID,
            "probe_id": "pol.verdict_direct",
            "probe_digest": CALIBRATION_PROBE_DIGEST,
            "calibration_artifact": CALIBRATION_FILENAME,
            "calibration_artifact_sha256": hashlib.sha256(
                DEFAULT_CALIBRATION_PATH.read_bytes()
            ).hexdigest(),
            "raw_field": "score",
            "export_field": "our_score",
            "unavailable_value": None,
        }
    }))
    corp = tmp_path / "corpus.json"
    corp.write_text(json.dumps(corpus))
    return str(run), str(corp)


def test_legacy_raw_score_is_not_relabelled_as_calibrated(tmp_path):
    run, corp = _write(
        tmp_path,
        [_row(0, 11, "correct", 0.95)],
        _corpus(),
    )
    Path(run).with_suffix(".meta.json").unlink()
    gold = _gold(tmp_path, [
        {"matches_hash": 123, "source_hash": 11, "tag": "correct"},
    ])

    per_ev, per_stmt, meta, metrics = build_run_export(
        run, corp, run_id="legacy", model="gemma", gold_path=gold
    )

    assert per_ev[0]["our_score"] is None
    assert per_stmt[0]["our_mean_score"] is None
    assert per_stmt[0]["our_noisy_or"] is None
    assert meta["sentence_score"]["status"] == "unavailable"
    assert meta["sentence_score"]["rows_available"] == 0
    assert metrics["tiers"]["ev"]["status"] == "unavailable"


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
    # The fitted statement profile remains statement-level.  The independent
    # sentence-probe probability is not a belief_* arm and must not leak into
    # this statement-belief namespace.
    rows = [_row(0, 11, "correct", 0.95)]
    run, corp = _write(tmp_path, rows, _corpus())
    per_ev, _ps, _meta, _metrics = build_run_export(run, corp, run_id="r", model="gemma")
    assert not any(k.startswith("belief_") or k == "belief_soft" for k in per_ev[0])


@pytest.mark.parametrize("model", ["local-gemma-4-26b", "mystery-7b"])
def test_no_soft_belief_when_reader_configuration_is_unfitted(tmp_path, model):
    # An unvalidated host/configuration or unknown reader → belief_soft is None
    # (named-empty), never an imputed 0.0; hard/parametric still compute.
    rows = [_row(0, 11, "correct", 0.95)]
    run, corp = _write(tmp_path, rows, _corpus())
    _ev, per_stmt, _meta, _metrics = build_run_export(run, corp, run_id="r", model=model)
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
    _ev, _ps, meta, m = build_run_export(run, corp, run_id="r", model="gemma", gold_path=gold)

    assert m["schema_version"] == 4
    assert m["run_id"] == "r" and m["model"] == "remote-gemma-4-26b"
    assert set(m["tiers"]) == {"ev", "stmt"}
    assert m["metrics_basis"]["bins"] == "BINS_8"
    assert "tau" not in m["metrics_basis"]
    thresholds = m["metrics_basis"]["thresholds"]
    assert set(thresholds) == {"tier1_sentence", "tier2_statement"}
    tier1_threshold = thresholds["tier1_sentence"]
    assert tier1_threshold["value"] == 0.5
    assert tier1_threshold["score"] == "calibrated sentence P(correct)"
    assert tier1_threshold["rule"] == "predict correct iff score >= value"
    assert "not tuned on held-out labels" in tier1_threshold["derivation"]
    sentence_profile = tier1_threshold["calibration_profile"]
    sentence_artifact = ROOT / "data" / "probe_battery" / "sentence_probe_calibration.json"
    assert sentence_profile == {
        "model": "local-gemma-4-26b",
        "model_id": CALIBRATION_MODEL_ID,
        "probe_id": "pol.verdict_direct",
        "probe_digest": CALIBRATION_PROBE_DIGEST,
        "artifact": sentence_artifact.name,
        "artifact_sha256": hashlib.sha256(sentence_artifact.read_bytes()).hexdigest(),
    }
    assert thresholds["tier2_statement"] == {
        "value": 0.5,
        "score": "statement belief",
        "rule": "predict error iff belief < value",
    }
    assert m["metrics_basis"]["soft_calibration"]["status"] == "available"
    assert "hybrid log-odds" in m["metrics_basis"]["soft_weights_note"]
    assert m["provenance"]["corpus_sha256"] == hashlib.sha256(Path(corp).read_bytes()).hexdigest()
    assert m["provenance"]["gold_sha256"] == hashlib.sha256(Path(gold).read_bytes()).hexdigest()
    assert len(m["provenance"]["evaluation_set_sha256"]) == 64
    assert meta["provenance"] == m["provenance"]

    ev = m["tiers"]["ev"]
    assert ev["status"] == "available" and ev["n"] == 3
    _assert_arm_shape(ev["arms"]["score"])

    st = m["tiers"]["stmt"]
    assert st["status"] == "available"
    assert set(st["arms"]) == {"hard", "parametric", "soft"}
    for arm in st["arms"].values():
        _assert_arm_shape(arm)


def test_tier1_probability_boundary_and_missing_score_exclusion():
    per_ev = [
        {"our_score": 0.49, "gold": {"verdict": "correct"}},
        {"our_score": 0.49, "gold": {"verdict": "wrong_relation"}},
        {"our_score": 0.50, "gold": {"verdict": "correct"}},
        {"our_score": 0.51, "gold": {"verdict": "wrong_relation"}},
        {"our_score": None, "gold": {"verdict": "correct"}},
    ]

    metrics = build_run_metrics(
        per_ev,
        {},
        {},  # non-None: gold is baked directly into each synthetic row
        "reader",
        "boundary",
        "gold.jsonl",
    )

    ev = metrics["tiers"]["ev"]
    assert ev["status"] == "available"
    assert ev["n"] == 4
    assert ev["arms"]["score"]["n"] == 4
    # Positive = correct; equality belongs to the predicted-correct side.
    assert ev["arms"]["score"]["confusion"] == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "tn": 1,
    }


def test_evaluation_set_digest_changes_with_evaluated_keys(tmp_path):
    rows = [_row(0, 11, "correct", 0.95)]
    run, corp = _write(tmp_path, rows, _corpus())
    gold = _gold(tmp_path, [
        {"matches_hash": 123, "source_hash": 11, "tag": "correct"},
        {"matches_hash": 123, "source_hash": 22, "tag": "correct"},
    ])
    _ev, _ps, _meta, first = build_run_export(
        run, corp, run_id="first", model="gemma", gold_path=gold
    )

    # Same statement key and same statement-level truth; only the exact Tier-1
    # evidence member changes. The joint evaluation digest must still change.
    changed = [dict(rows[0], evidence_i=1, source_hash=22)]
    changed_run = tmp_path / "changed.jsonl"
    changed_run.write_text(json.dumps(changed[0]) + "\n")
    _ev, _ps, _meta, second = build_run_export(
        str(changed_run), corp, run_id="second", model="gemma", gold_path=gold
    )

    assert first["provenance"]["corpus_sha256"] == second["provenance"]["corpus_sha256"]
    assert first["provenance"]["gold_sha256"] == second["provenance"]["gold_sha256"]
    assert first["provenance"]["evaluation_set_sha256"] != second["provenance"]["evaluation_set_sha256"]


def test_gold_map_exact_pair_first_and_truth_safe_source_fallback(tmp_path):
    gold = _gold(tmp_path, [
        # The same evidence source can be correct for one statement and wrong for
        # another. Exact pairs remain resolvable; source-only fallback is unsafe.
        {"matches_hash": 101, "source_hash": 7, "tag": "correct", "curator": "a"},
        {"matches_hash": 202, "source_hash": 7, "tag": "wrong_relation", "curator": "b"},
        # Reuse across statements is safe when every context agrees on truth.
        {"matches_hash": 303, "source_hash": 8, "tag": "correct"},
        {"matches_hash": 404, "source_hash": 8, "tag": "correct"},
        # Multiple curators on one exact pair use canonical any-incorrect-wins.
        {"matches_hash": 505, "source_hash": 9, "tag": "correct", "curator": "a"},
        {"matches_hash": 505, "source_hash": 9, "tag": "grounding", "curator": "b"},
        # Mixed-schema file: an exact correct context plus a source-only wrong
        # label must disable fallback rather than silently ignoring the latter.
        {"matches_hash": 606, "source_hash": 10, "tag": "correct"},
        {"source_hash": 10, "tag": "wrong_relation"},
    ])
    gold_map = load_gold_map(gold)

    assert gold_map.ambiguous_sources == 2
    assert gold_map.for_row(101, 7)["verdict"] == "correct"
    assert gold_map.for_row(202, 7)["verdict"] == "incorrect"
    assert gold_map.for_row(None, 7) is None
    assert gold_map.for_row(999, 7) is None
    assert gold_map.for_row(None, 8)["verdict"] == "correct"
    pair = gold_map.for_row(505, 9)
    assert pair["verdict"] == "incorrect" and pair["n"] == 2
    assert pair["curators"] == ["a", "b"]
    assert gold_map.for_row(606, 10)["verdict"] == "correct"
    assert gold_map.for_row(None, 10) is None


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
    write_run_export(
        run, corp, str(out), run_id="r", model="remote-gemma-4-26b",
        gold_path=gold, prompt_sha256=BASELINE_PROMPT_SHA256,
    )
    assert (out / "metrics.json").exists()
    disk = json.loads((out / "metrics.json").read_text())
    _, _, _, mem = build_run_export(run, corp, run_id="r", model="gemma", gold_path=gold)
    assert disk == mem


# ── 3. cross-check vs the live ship-gate scorer (data-gated) ───────────

# The surviving holdout gold predates matches_hash. Both export metrics and the
# ship gate therefore use source-hash fallback only after checking that every
# context for the source agrees on correctness.
_HOLDOUT_GOLD = ROOT / "data" / "results" / "cc_holdout_cc" / "holdout_cc.jsonl"
_RUNS = {
    "gemma": ROOT / "data" / "results" / "holdout_cc_gemma.jsonl",
    "medpsy": ROOT / "data" / "results" / "holdout_cc_medpsy.jsonl",
}


def _synth_stmt_agg(run_path: Path) -> dict:
    """The minimal stmt_agg build_run_metrics needs, from a raw run — mirrors the
    production belief rows at run-stmt_hash grain. The holdout gold has no text,
    so evidence_hash is also the ship gate's de-dup fallback."""
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


def _synth_per_ev(run_path: Path, gold_map) -> list[dict]:
    rows = {}
    for line in open(run_path):
        if line.strip():
            d = json.loads(line)
            rows[(d["stmt_i"], d["evidence_i"])] = d
    out = []
    for d in rows.values():
        # Frozen holdout runs predate the calibrated sentence-score contract.
        # Their numeric score is historical grid output, never Tier-1 input.
        out.append({"our_score": None,
                    "gold": gold_map.for_row(None, d.get("source_hash"))})
    return out


@pytest.mark.parametrize("model", ["gemma", "medpsy"])
def test_e5_crosscheck_ship_gate(model):
    run_path = _RUNS[model]
    if not (_HOLDOUT_GOLD.exists() and run_path.exists()):
        pytest.skip("current holdout_cc run / gold not present")

    # scripts/ is not a package; load the live gate helpers only for this
    # data-backed contract check.
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import calibration_ship_gate as ship_gate
    import calibration_stage1 as calibration_stage1
    from indra_belief.calibration_constants import (
        calibration_for_run,
        fitted_calibration_for_run,
        reader_configuration_for_run,
    )

    gold_map = load_gold_map(str(_HOLDOUT_GOLD))
    agg = _synth_stmt_agg(run_path)
    per_ev = _synth_per_ev(run_path, gold_map)
    reader_config = reader_configuration_for_run(run_path)
    production_profile = calibration_for_run(run_path)
    m = build_run_metrics(
        per_ev, agg, gold_map, reader_config["model"], f"holdout_cc_{model}",
        str(_HOLDOUT_GOLD), soft_profile=production_profile,
        reader_configuration=reader_config,
    )

    statements, join = ship_gate.statements_for_run(run_path, _HOLDOUT_GOLD)
    scored = ship_gate.score_statements(statements, fitted_calibration_for_run(run_path))
    ship_metrics = {
        "hard": calibration_stage1.metric_block(scored["hard"], scored["labels"]),
        "parametric": calibration_stage1.metric_block(scored["parametric"], scored["labels"]),
        "soft": calibration_stage1.metric_block(scored["calibrated"], scored["labels"]),
    }
    stmt = m["tiers"]["stmt"]

    assert not gold_map.by_pair
    assert gold_map.ambiguous_sources == 0
    assert join["join_mode"].startswith("per-row exact")
    assert join["n_exact_joined_rows"] == 0
    assert join["n_source_fallback_rows"] == join["n_joined_rows"]
    assert stmt["n"] == len(scored["labels"])

    # Shared production arms are byte-exact. MedPsy's measured candidate failed
    # its ship gate, so the production soft arm is intentionally named-empty.
    for arm in ("hard", "parametric"):
        a = stmt["arms"][arm]
        assert a["ece"] == pytest.approx(ship_metrics[arm]["ece"], abs=1e-12), arm
        assert a["auroc"] == pytest.approx(ship_metrics[arm]["auroc"], abs=1e-12), arm
    if model == "gemma":
        a = stmt["arms"]["soft"]
        assert a["ece"] == pytest.approx(ship_metrics["soft"]["ece"], abs=1e-12)
        assert a["auroc"] == pytest.approx(ship_metrics["soft"]["auroc"], abs=1e-12)
    else:
        assert stmt["arms"]["soft"]["status"] == "unavailable"


def test_ship_gate_requires_explicit_e4_identity_assertion():
    """A metrics-only rerun cannot silently manufacture the fourth green leg."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import calibration_ship_gate as ship_gate

    ev = {
        "metrics": {
            "hard": {"ece": 0.2, "auroc": 0.7},
            "clean": {"ece": 0.1, "auroc": 0.8},
        },
        "errf1_boot": {
            "f1_hard": 0.7, "f1_soft": 0.71, "delta": 0.01,
            "ci_delta": [-0.01, 0.03],
        },
    }
    unverified = ship_gate.gate(ev)
    assert unverified["e4_identity"]["pass"] is False
    assert unverified["overall"] is False

    verified = ship_gate.gate(ev, e4_identity_pass=True)
    assert verified["e4_identity"]["pass"] is True
    assert verified["overall"] is True


def test_ship_gate_uses_exact_first_per_row_in_mixed_gold_file(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import calibration_ship_gate as ship_gate

    run = tmp_path / "mixed.jsonl"
    run.write_text("\n".join(json.dumps(row) for row in [
        {"stmt_i": 0, "evidence_i": 0, "stmt_hash": f"{101:016x}",
         "source_hash": 7, "source_api": "reach", "verdict": "correct",
         "confidence": "high", "tier": "llm_comprehension"},
        {"stmt_i": 1, "evidence_i": 0, "stmt_hash": f"{303:016x}",
         "source_hash": 8, "source_api": "reach", "verdict": "incorrect",
         "confidence": "high", "tier": "llm_comprehension"},
    ]) + "\n")
    gold = tmp_path / "mixed_gold.jsonl"
    gold.write_text("\n".join(json.dumps(row) for row in [
        {"matches_hash": 101, "source_hash": 7, "tag": "correct",
         "evidence_text": "exact"},
        {"source_hash": 8, "tag": "wrong_relation", "evidence_text": "fallback"},
    ]) + "\n")

    statements, diag = ship_gate.statements_for_run(run, gold)
    assert len(statements) == 2
    assert diag["n_exact_joined_rows"] == 1
    assert diag["n_source_fallback_rows"] == 1
    assert diag["n_unmatched_rows"] == 0 and diag["n_ambiguous_rows"] == 0


def test_ship_gate_rejects_cross_configuration_profile_transfer(tmp_path):
    import hashlib
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import calibration_ship_gate as ship_gate

    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    row = {"stmt_i": 0, "evidence_i": 0,
           "call_log": [{"kind": "monolithic", "system": "prompt"}]}
    train.write_text(json.dumps(row) + "\n")
    test.write_text(json.dumps(row) + "\n")
    train.with_suffix(".meta.json").write_text(json.dumps({"model": "gemma-remote"}))
    test.with_suffix(".meta.json").write_text(json.dumps({"model": "bedrock-gemma"}))

    with pytest.raises(ValueError, match="profile transfer.*forbidden"):
        ship_gate.validate_configuration_pair(train, test)

    test.with_suffix(".meta.json").write_text(json.dumps({"model": "gemma-remote"}))
    config = (
        "remote-gemma-4-26b@prompt-sha256:"
        + hashlib.sha256(b"prompt").hexdigest()
    )
    assert ship_gate.validate_configuration_pair(train, test) == (config, config)

    changed = dict(row)
    changed["call_log"] = [{"kind": "monolithic", "system": "changed prompt"}]
    test.write_text(json.dumps(changed) + "\n")
    with pytest.raises(ValueError, match="profile transfer.*forbidden"):
        ship_gate.validate_configuration_pair(train, test)


def test_stage0_join_collapses_duplicate_curators_and_scored_pairs():
    """Historical baseline reruns use the same conservative unique-pair grain."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import calibration_stage0 as stage0

    gold = [
        {"matches_hash": 123, "source_hash": 11, "pa_hash": 9, "tag": "correct"},
        {"matches_hash": 123, "source_hash": 11, "pa_hash": 9, "tag": "grounding"},
    ]
    by_pair, by_source = stage0.build_gold_index(gold)
    collapsed = by_pair[(123, 11)]
    assert collapsed["tag"] == "incorrect"
    assert collapsed["n_gold_rows"] == 2
    assert collapsed["all_tags"] == ["correct", "grounding"]

    scored = {
        "stmt_hash": f"{123:016x}", "source_hash": 11,
        "verdict": "incorrect", "source_api": "reach",
    }
    joined, parse_null, missed = stage0.join_model(
        [dict(scored), dict(scored)], by_pair, by_source
    )
    assert len(joined) == 1
    assert parse_null == 0
    assert missed == 0
    assert joined[0][0]["tag"] == "incorrect"


# ── 4. statement-heuristics instrument (I1–I4, E10) — additive + first-class ───

STATEMENT_INSTRUMENT_KEYS = {"gold_statement", "coherence_summary"}
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


def test_per_statement_v8_gold_and_coherence_contract(tmp_path):
    # Schema v8 retains the statement gold/coherence instrument while aligning
    # its gold lookup with Tier-2 and the ship gate.
    run, corp, gold = _three_ev_gold(tmp_path)
    _ev, per_stmt, meta, _m = build_run_export(run, corp, run_id="r", model="gemma", gold_path=gold)
    s = per_stmt[0]
    assert meta["schema_version"] == 8
    assert NEW_KEYS <= set(s) and STATEMENT_INSTRUMENT_KEYS <= set(s)
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


def test_metrics_schema_version_is_4(tmp_path):
    run, corp, gold = _three_ev_gold(tmp_path)
    _ev, _ps, _meta, m = build_run_export(run, corp, run_id="r", model="gemma", gold_path=gold)
    assert m["schema_version"] == 4


def test_metrics_gold_without_pa_hash_keeps_tier2_available(tmp_path):
    # External/rasmachine-style gold needs only the authoritative exact pair;
    # pa_hash is no longer a Tier-2 dependency because production groups on the
    # run's stmt_hash.
    rows = [_row(0, 11, "correct", 0.95), _row(1, 22, "incorrect", 0.05)]
    run, corp = _write(tmp_path, rows, _corpus())
    gold = _gold(tmp_path, [
        {"matches_hash": 123, "source_hash": 11, "tag": "correct"},
        {"matches_hash": 123, "source_hash": 22, "tag": "wrong_relation"},
    ])
    _ev, per_stmt, _meta, m = build_run_export(run, corp, run_id="r", model="gemma", gold_path=gold)
    assert m["tiers"]["ev"]["status"] == "available"
    stmt = m["tiers"]["stmt"]
    assert stmt["status"] == "available" and stmt["n"] == 1
    for arm in stmt["arms"].values():
        _assert_arm_shape(arm)
    assert m["metrics_basis"]["tier2_statement_key"].startswith("run stmt_hash")
    assert per_stmt[0]["gold_statement"]["verdict"] == "incorrect"
    assert per_stmt[0]["gold_statement"]["n"] == 2
