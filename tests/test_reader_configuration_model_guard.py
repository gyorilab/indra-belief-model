"""The reader-configuration resolver must verify the MODEL, not just the prompt.

``gemma_remote`` and ``medpsy_remote`` are both keyed on BASELINE_PROMPT_SHA256, so
the prompt fingerprint cannot separate them: a mislabelled/template-copied
``.meta.json`` alone used to be enough to hand a MedPsy-served run the ENABLED gemma
profile even though MedPsy's own profile failed its ship gate. These tests pin the
served-model cross-check that closes that asymmetry — and, just as importantly, pin
the cases where the guard must stay silent, because absence of a recorded served id
is not evidence of a mismatch.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from indra_belief.calibration_constants import (
    BASELINE_PROMPT_SHA256,
    calibration_for,
    calibration_for_run,
    fitted_calibration_for,
    fitted_calibration_for_run,
    model_fingerprints_for_run,
    prompt_fingerprints_for_run,
    reader_configuration_for_run,
)
from indra_belief.results import _soft_calibration_block

ROOT = Path(__file__).resolve().parents[1]
GEMMA_FIT_RUN = ROOT / "data/results/eval_curation_v1_gemma.jsonl"
MEDPSY_FIT_RUN = ROOT / "data/results/eval_curation_v1_medpsy.jsonl"

SYSTEM = "You judge whether a biomedical text-mining extraction is correct.\n"
SYSTEM_SHA256 = hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest()


def _write_run(tmp_path: Path, calls: list[list[dict]], meta: dict) -> Path:
    """Synthesize a minimal run + its .meta.json sidecar."""
    run = tmp_path / "synthetic_run.jsonl"
    with run.open("w") as fh:
        for i, call_log in enumerate(calls):
            fh.write(json.dumps({"stmt_i": i, "evidence_i": 0, "call_log": call_log}) + "\n")
    run.with_suffix(".meta.json").write_text(json.dumps(meta))
    return run


def _monolithic(model_id: str | None = None) -> dict:
    call = {"kind": "monolithic", "system": SYSTEM}
    if model_id is not None:
        call["model_id"] = model_id
    return call


# ── honest label: the shipped resolution is untouched ────────────────────────

def test_honest_fit_run_still_resolves_to_its_shipped_profile():
    if not GEMMA_FIT_RUN.exists():
        pytest.skip("gemma fit run not present")
    config = reader_configuration_for_run(GEMMA_FIT_RUN)
    assert config["status"] == "identified"
    assert config["model"] == "remote-gemma-4-26b"
    assert config["prompt_sha256"] == BASELINE_PROMPT_SHA256
    assert config["id"] == f"remote-gemma-4-26b@prompt-sha256:{BASELINE_PROMPT_SHA256}"
    # the run's OWN call log names the gateway's served id — the guard's evidence
    assert set(config["model_fingerprints"]) == {"gemma-4-26b-ollama"}
    profile = calibration_for_run(GEMMA_FIT_RUN)
    assert profile is not None
    assert profile["profile_id"] == "remote-gemma-4-26b@prompt-b44638216740@eval_curation_v1"


# ── mislabelled meta: the enabled profile is no longer reachable ─────────────

def test_medpsy_run_labelled_gemma_is_a_model_mismatch():
    """The reproduction case: MedPsy-served rows under a gemma label.

    Both profiles share BASELINE_PROMPT_SHA256, so the prompt fingerprint agrees
    and cannot catch this. Before the served-model cross-check this resolved to the
    ENABLED gemma profile (log_lr_confirm 1.503).
    """
    if not MEDPSY_FIT_RUN.exists():
        pytest.skip("medpsy fit run not present")
    config = reader_configuration_for_run(MEDPSY_FIT_RUN, "gemma-remote")
    assert config["status"] == "mismatch"
    assert config["id"] is None
    # nulling the prompt is what actually gates: the calibration resolvers key on
    # (model, prompt_sha256) and never read ``id``
    assert config["prompt_sha256"] is None
    # the declared label is retained — results.build_run_metrics threads it through
    assert config["model"] == "remote-gemma-4-26b"
    assert set(config["model_fingerprints"]) == {"medpsy-4b"}
    assert calibration_for_run(MEDPSY_FIT_RUN, "gemma-remote") is None
    assert fitted_calibration_for_run(MEDPSY_FIT_RUN, "gemma-remote") is None


def test_medpsy_run_under_its_own_honest_label_is_unaffected():
    if not MEDPSY_FIT_RUN.exists():
        pytest.skip("medpsy fit run not present")
    config = reader_configuration_for_run(MEDPSY_FIT_RUN)
    assert config["status"] == "identified"
    assert config["model"] == "remote-medpsy-4b"
    fitted = fitted_calibration_for_run(MEDPSY_FIT_RUN)
    assert fitted["profile_id"] == "remote-medpsy-4b@prompt-b44638216740@eval_curation_v1"
    # measured but ship-gate-disabled: production still resolves to None
    assert calibration_for_run(MEDPSY_FIT_RUN) is None


# ── absence is never evidence ────────────────────────────────────────────────

def test_empty_call_log_resolves_exactly_as_before(tmp_path):
    """Rows handled without an LLM call make no claim about the served model."""
    run = _write_run(
        tmp_path, [[], []],
        {"model": "gemma-remote", "prompt_sha256": BASELINE_PROMPT_SHA256},
    )
    config = reader_configuration_for_run(run)
    assert config == {
        "status": "identified",
        "id": f"remote-gemma-4-26b@prompt-sha256:{BASELINE_PROMPT_SHA256}",
        "model": "remote-gemma-4-26b",
        "prompt_sha256": BASELINE_PROMPT_SHA256,
        "prompt_fingerprint_source": "run_metadata",
        "declared_prompt_sha256": BASELINE_PROMPT_SHA256,
        "prompt_fingerprints": {},
        "model_fingerprints": {},
    }
    assert calibration_for_run(run) is not None


def test_call_rows_without_model_id_resolve_exactly_as_before(tmp_path):
    """Older decomposed-phase runs logged calls with no ``model_id`` at all."""
    run = _write_run(tmp_path, [[_monolithic()]], {"model": "gemma-remote"})
    config = reader_configuration_for_run(run)
    assert config["status"] == "identified"
    assert config["model_fingerprints"] == {}
    assert config["prompt_sha256"] == SYSTEM_SHA256
    assert config["id"] == f"remote-gemma-4-26b@prompt-sha256:{SYSTEM_SHA256}"


def test_unknown_declared_model_has_no_expectation_on_record(tmp_path):
    """A name absent from the registry has no accepted-id expectation, so a
    single served id cannot CONTRADICT it. (It is not exempt from the ambiguity
    branch — see test_unknown_declared_model_is_still_refused_when_ids_differ.)"""
    run = _write_run(
        tmp_path, [[_monolithic("some-served-id")]],
        {"model": "not-a-registered-model"},
    )
    config = reader_configuration_for_run(run)
    assert config["status"] == "identified"
    assert config["model_fingerprints"] == {"some-served-id": 1}


# ── historical served id: shipped provenance keeps resolving ────────────────

def test_pre_rename_served_id_is_accepted_for_remote_gemma(tmp_path):
    """5e89e2c renamed the gateway id gemma-4-26b -> gemma-4-26b-ollama.

    Runs exported before that rename (rasmachine_mono_gemma_remote_direct) are
    honestly labelled and already shipped; the historical id must stay accepted.
    """
    run = _write_run(
        tmp_path, [[_monolithic("gemma-4-26b")]],
        {"model": "gemma-remote", "prompt_sha256": SYSTEM_SHA256},
    )
    config = reader_configuration_for_run(run)
    assert config["status"] == "identified"
    assert config["id"] == f"remote-gemma-4-26b@prompt-sha256:{SYSTEM_SHA256}"


def test_shipped_pre_rename_run_still_resolves_to_the_gemma_profile():
    run = ROOT / "data/results/rasmachine_mono_gemma_remote_direct.jsonl"
    if not run.exists():
        pytest.skip("pre-rename rasmachine run not present")
    config = reader_configuration_for_run(run)
    assert config["status"] == "identified"
    assert set(config["model_fingerprints"]) == {"gemma-4-26b"}
    profile = calibration_for_run(run)
    assert profile["profile_id"] == "remote-gemma-4-26b@prompt-b44638216740@eval_curation_v1"


def test_historical_alias_does_not_leak_to_other_models(tmp_path):
    """The alias is provenance for ONE configuration, not a family-wide widening."""
    run = _write_run(
        tmp_path, [[_monolithic("gemma-4-26b")]],
        {"model": "medpsy-remote", "prompt_sha256": SYSTEM_SHA256},
    )
    config = reader_configuration_for_run(run)
    assert config["status"] == "mismatch"
    assert config["id"] is None
    assert config["prompt_sha256"] is None


# ── more than one served model in one run ───────────────────────────────────

def test_two_distinct_served_models_are_mixed(tmp_path):
    run = _write_run(
        tmp_path,
        [[_monolithic("gemma-4-26b-ollama")], [_monolithic("medpsy-4b")]],
        {"model": "gemma-remote", "prompt_sha256": SYSTEM_SHA256},
    )
    config = reader_configuration_for_run(run)
    assert config["status"] == "mixed"
    assert config["id"] is None
    assert config["prompt_sha256"] is None
    assert config["model_fingerprints"] == {"gemma-4-26b-ollama": 1, "medpsy-4b": 1}
    assert calibration_for_run(run) is None


def test_prompt_disagreement_keeps_its_own_more_specific_status(tmp_path):
    """A prompt mismatch already hard-gates; the model check must not overwrite it."""
    run = _write_run(
        tmp_path, [[_monolithic("medpsy-4b")]],
        {"model": "gemma-remote", "prompt_sha256": BASELINE_PROMPT_SHA256},
    )
    config = reader_configuration_for_run(run)
    assert config["status"] == "mismatch"
    assert config["declared_prompt_sha256"] == BASELINE_PROMPT_SHA256
    assert config["prompt_sha256"] is None


# ── the fingerprint twin's own contract ─────────────────────────────────────

def test_model_fingerprints_are_sorted_and_counted(tmp_path):
    run = _write_run(
        tmp_path,
        [[_monolithic("z-served"), _monolithic("a-served")], [_monolithic("z-served")]],
        {"model": "gemma-remote"},
    )
    counts = model_fingerprints_for_run(run)
    assert counts == {"a-served": 1, "z-served": 2}
    assert list(counts) == ["a-served", "z-served"]


@pytest.mark.parametrize("name", ["absent.jsonl", "corrupt.jsonl", "adir"])
def test_model_fingerprints_tolerate_missing_and_corrupt_inputs(tmp_path, name):
    target = tmp_path / name
    if name == "corrupt.jsonl":
        target.write_text("{not json at all\n")
    elif name == "adir":
        target.mkdir()
    assert model_fingerprints_for_run(target) == {}
    # the prompt twin tolerates exactly the same inputs — one shared walker
    assert prompt_fingerprints_for_run(target) == {}


def test_blank_and_whitespace_served_ids_are_not_observations(tmp_path):
    null_id = {"kind": "monolithic", "system": SYSTEM, "model_id": None}
    run = _write_run(
        tmp_path,
        [[_monolithic("   "), _monolithic(""), null_id]],
        {"model": "gemma-remote"},
    )
    config = reader_configuration_for_run(run)
    assert config["model_fingerprints"] == {}
    assert config["status"] == "identified"


def test_unknown_declared_model_is_still_refused_when_ids_differ(tmp_path):
    """Absence of an expectation is not a licence to serve two endpoints.

    The ambiguity branch fires BEFORE the accepted-id check, so an unregistered
    declared name still yields 'mixed': "which endpoint produced this run" has no
    answer at all, independent of what the run claims to be.
    """
    run = _write_run(
        tmp_path,
        [[_monolithic("some-served-id")], [_monolithic("another-served-id")]],
        {"model": "not-a-registered-model", "prompt_sha256": SYSTEM_SHA256},
    )
    config = reader_configuration_for_run(run)
    assert config["status"] == "mixed"
    assert config["id"] is None
    assert config["prompt_sha256"] is None
    assert calibration_for_run(run) is None


# ── precedence: a PROMPT status wins over a model status ─────────────────────
#
# calibration_constants gates the model cross-check behind ``status ==
# "identified"``. Both directions below are discriminating: flipping that guard
# to a bare ``if model_status:`` makes exactly these two tests fail, on the
# status AND on the sentence results.py renders.


def _reason(config: dict) -> str:
    """The sentence _soft_calibration_block bakes for an unavailable profile."""
    block = _soft_calibration_block(
        config["model"], config,
        calibration_for(config["model"], prompt_sha256=config["prompt_sha256"]),
        fitted_calibration_for(config["model"], prompt_sha256=config["prompt_sha256"]),
    )
    assert block["status"] == "unavailable"
    return block["reason"]


def test_prompt_mixed_outranks_a_contradicting_served_id(tmp_path):
    """Two monolithic prompts (prompt 'mixed') crossed with a model MISMATCH."""
    other = {"kind": "monolithic", "system": SYSTEM + "Answer in JSON.\n",
             "model_id": "medpsy-4b"}
    run = _write_run(
        tmp_path, [[_monolithic("medpsy-4b")], [other]],
        {"model": "gemma-remote", "prompt_sha256": SYSTEM_SHA256},
    )
    config = reader_configuration_for_run(run)
    # the model half would say 'mismatch'; the prompt half is more specific
    assert config["status"] == "mixed"
    assert len(config["prompt_fingerprints"]) == 2
    assert set(config["model_fingerprints"]) == {"medpsy-4b"}
    assert _reason(config) == "run contains more than one monolithic prompt fingerprint"
    assert calibration_for_run(run) is None


def test_prompt_mismatch_outranks_two_served_ids(tmp_path):
    """Declared prompt != the single observed one, crossed with a model 'mixed'."""
    run = _write_run(
        tmp_path,
        [[_monolithic("gemma-4-26b-ollama")], [_monolithic("medpsy-4b")]],
        {"model": "gemma-remote", "prompt_sha256": BASELINE_PROMPT_SHA256},
    )
    config = reader_configuration_for_run(run)
    # the model half would say 'mixed'; the prompt half is more specific
    assert config["status"] == "mismatch"
    assert list(config["prompt_fingerprints"]) == [SYSTEM_SHA256]
    assert len(config["model_fingerprints"]) == 2
    assert _reason(config) == "declared prompt fingerprint disagrees with persisted call logs"
    assert calibration_for_run(run) is None


# ── scope: a served id on ANY call kind is reader configuration ──────────────
#
# The decision and its rationale live at the counting site in
# calibration_constants._call_log_fingerprints. These pin it: the belief scalar
# comes out of the whole call chain, so a heterogeneous-endpoint run is refused,
# and a decomposed run with no monolithic call is still guarded.


def _sub_call(kind: str, model_id: str) -> dict:
    """A non-scoring phase call: no monolithic ``system``, but a served id."""
    return {"kind": kind, "model_id": model_id}


def test_served_id_on_a_non_monolithic_call_still_guards(tmp_path):
    """A decomposed run declares its prompt in metadata and runs no monolithic
    call, so the prompt half resolves 'identified' and would hand over a profile.
    The only evidence that the endpoint is wrong sits on probe calls."""
    run = _write_run(
        tmp_path,
        [[_sub_call("probe_subject_role", "medpsy-4b"),
          _sub_call("verify_grounding", "medpsy-4b")]],
        {"model": "gemma-remote", "prompt_sha256": BASELINE_PROMPT_SHA256},
    )
    config = reader_configuration_for_run(run)
    assert config["status"] == "mismatch"
    assert config["id"] is None
    assert config["prompt_sha256"] is None
    assert config["prompt_fingerprints"] == {}
    assert config["model_fingerprints"] == {"medpsy-4b": 2}
    assert calibration_for_run(run) is None


def test_monolithic_and_sub_call_on_different_endpoints_is_mixed(tmp_path):
    """Half the chain on gemma, the adjudicating half on medpsy: no single
    fitted endpoint produced this run's scalar, so no profile may claim it."""
    run = _write_run(
        tmp_path,
        [[_monolithic("gemma-4-26b-ollama"), _sub_call("relation_nature", "medpsy-4b")]],
        {"model": "gemma-remote", "prompt_sha256": SYSTEM_SHA256},
    )
    config = reader_configuration_for_run(run)
    assert config["status"] == "mixed"
    assert config["id"] is None
    assert config["prompt_sha256"] is None
    assert config["model_fingerprints"] == {"gemma-4-26b-ollama": 1, "medpsy-4b": 1}
    assert calibration_for_run(run) is None
    assert fitted_calibration_for_run(run) is None


def test_shipped_probe_only_run_resolution_is_unchanged():
    """The one shipped run whose served ids appear ONLY on non-monolithic kinds.

    It records no monolithic prompt at all, so it already sits on the hard gate
    via 'missing_prompt' and the served-id scope moves nothing for it. This is
    the regression pin for the all-kinds decision on real data.
    """
    run = ROOT / "data/results/eval_curation_v1_medpsy_decomp.jsonl"
    if not run.exists():
        pytest.skip("decomposed medpsy run not present")
    config = reader_configuration_for_run(run)
    assert config["status"] == "missing_prompt"
    assert config["model"] is None            # the run has no .meta.json sidecar
    assert config["prompt_fingerprints"] == {}
    assert config["model_fingerprints"] == {"medpsy-4b": 8851}
    assert calibration_for_run(run) is None
