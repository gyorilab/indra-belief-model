"""Per-reader belief calibration from a measured confusion matrix.

There are no hand-set *reader* weights here. Each reader **configuration** is
characterized by its confusion matrix on gold — the model's verdict crossed with
the curator's label, tallied on unique evidence pairs in eval_curation_v1:

    cc = confirmed & correct      ci = confirmed & incorrect
    ic = rejected  & correct      ii = rejected  & incorrect

Everything the belief model uses is *derived* from these counts, not assigned:

  * a verdict's measured accuracy    P(correct | verdict) = right / (right + wrong)
    — the two numbers the reliability slide shows.
  * a verdict's weight of evidence: the log-likelihood ratio
    ``log(P(verdict | correct) / P(verdict | incorrect))``.

The statement model averages repeated measurements within a source, sums source
contributions in log-odds space, and applies a sigmoid. A confirmed read also has
a conservative source-reliability floor derived from the separately fitted INDRA
source priors: its contribution is the larger of the reader's measured confirm
log-LR and the source reliability log-odds. This is an explicit hybrid heuristic,
not a pure Bayesian posterior and not another reader-fit parameter. Rejections use
the reader's measured reject log-LR. At the fit prior, a single ordinary source
whose floor does not bind reduces to the observed ``P(correct | verdict)``.

The counts are the only *reader-profile* fit data; the hybrid source floor also
uses ``RECALIBRATED_PRIORS`` from a separate 9,342-curation source fit. Profiles
resolve by exact serving/scorer configuration and travel with the run. Same model
weights on a different host or reasoning mode do not inherit a profile. An
unfitted configuration resolves to None and stays on the hard gate.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from .model_client import LOCAL_MODELS, canonical_model_name

# A reader fit is scoped to both the served model and the scoring prompt.  These
# are SHA-256 hashes of the exact monolithic ``system`` strings persisted in the
# fit-run call logs.  The full digest is pinned; prefixes are display-only.
BASELINE_PROMPT_SHA256 = "b4463821674084172f5f7237aa3e91048f8a57b32bd68e79bfe7a8aaf43f4581"
REASONING_FIRST_PROMPT_SHA256 = "07377e338ff2835fbb7cc5e714f047db7cfca1b76ed05e98622752d99fa1d364"
FIT_GOLD_SHA256 = "8e266acefd191e25a92f88febcb6f6d7f1b3be8c8d8f45a18012f76d9930f600"
HOLDOUT_GOLD_SHA256 = "aa022aa0d2543f7031a686ec661a3bc3f59dec7cb9cc12f049ff0068653ecb49"
EXTERNAL_GOLD_SHA256 = "52cde61f8f3e3dac01ad13f09c9d6db623eea888ffd617410d1c88de6527c80f"

# Reader configuration -> confusion matrix (verdict × curator gold) tallied on
# eval_curation_v1 after exact-pair multi-curator aggregation and duplicate-pair
# removal (n=1604 unique pairs). These four counts are the reader calibration.
_CONFUSION: dict[str, dict[str, int]] = {
    "gemma_remote": {"cc": 704, "ci": 157, "ic": 97, "ii": 646},
    "gemma_bedrock_rf": {"cc": 662, "ci": 81, "ic": 139, "ii": 722},
    "medpsy_remote": {"cc": 718, "ci": 230, "ic": 83, "ii": 573},
    # Self-hosted MLX, fitted 2026-08-13 on the same eval_curation_v1 protocol.
    # Same weights as gemma_bedrock_rf, different serving stack: the counts land
    # close (651/91/148/710 against 662/81/139/722), and a paired evidence-grain
    # comparison on the external gold put the two readers within noise of each
    # other (delta err-F1 -0.0082, 95% CI [-0.0259, +0.0092], 95.4% verdict
    # agreement over 560 shared pairs).
    "local_gemma_mlx": {"cc": 651, "ci": 91, "ic": 148, "ii": 710},
}

_PROFILE_META = {
    "gemma_remote": {
        "profile_id": "remote-gemma-4-26b@prompt-b44638216740@eval_curation_v1",
        "reader_model": "remote-gemma-4-26b",
        "prompt_sha256": BASELINE_PROMPT_SHA256,
        "fit_run": "data/results/eval_curation_v1_gemma.jsonl",
        "deployment_status": "enabled",
        "validation": {
            "result": "pass",
            "gold": "data/results/cc_holdout_cc/holdout_cc.jsonl",
            "gold_sha256": HOLDOUT_GOLD_SHA256,
            "run": "data/results/holdout_cc_gemma.jsonl",
            "gate": "4/4",
        },
    },
    "gemma_bedrock_rf": {
        "profile_id": "bedrock-gemma-4-26b@prompt-07377e338ff2@eval_curation_v1",
        "reader_model": "bedrock-gemma-4-26b",
        "prompt_sha256": REASONING_FIRST_PROMPT_SHA256,
        "fit_run": "data/results/eval_curation_v1_gemma_rf_bedrock.jsonl",
        "deployment_status": "enabled",
        "validation": {
            "result": "pass",
            "gold": "data/benchmark/external_curator_gold_v1.jsonl",
            "gold_sha256": EXTERNAL_GOLD_SHA256,
            "run": "data/results/external_curator_v1_bedrock-gemma.jsonl",
            "gate": "4/4",
        },
    },
    "medpsy_remote": {
        "profile_id": "remote-medpsy-4b@prompt-b44638216740@eval_curation_v1",
        "reader_model": "remote-medpsy-4b",
        "prompt_sha256": BASELINE_PROMPT_SHA256,
        "fit_run": "data/results/eval_curation_v1_medpsy.jsonl",
        "deployment_status": "disabled",
        "validation": {
            "result": "fail",
            "gold": "data/results/cc_holdout_cc/holdout_cc.jsonl",
            "gold_sha256": HOLDOUT_GOLD_SHA256,
            "run": "data/results/holdout_cc_medpsy.jsonl",
            "gate": "3/4 (ECE worsened)",
            "note": ("the external MedPsy run used prompt 07377e338ff2, so it "
                     "cannot validate this b44638216740 profile"),
        },
    },
    "local_gemma_mlx": {
        "profile_id": "local-gemma-4-26b@prompt-07377e338ff2@eval_curation_v1",
        "reader_model": "local-gemma-4-26b",
        "prompt_sha256": REASONING_FIRST_PROMPT_SHA256,
        "fit_run": "data/results/eval_curation_v1_local-gemma-4-26b.jsonl",
        "deployment_status": "enabled",
        "validation": {
            "result": "pass",
            "gold": "data/benchmark/external_curator_gold_v1.jsonl",
            "gold_sha256": EXTERNAL_GOLD_SHA256,
            "run": "data/results/external_curator_v1_local-gemma-4-26b.jsonl",
            "gate": "4/4",
            "note": ("ECE 0.231 -> 0.052, AUROC 0.793 -> 0.808, err-F1 0.796 -> "
                     "0.800 (delta +0.004, CI [-0.009, +0.019], non-inferior). "
                     "Both runs served by mlx_lm at max_tokens 8192; 4 capped "
                     "reads in the fit run were withheld rather than scored, so "
                     "no mid-thought verdict reached these counts."),
        },
    },
}

_FITTED_CONFIGS = {
    ("remote-gemma-4-26b", BASELINE_PROMPT_SHA256): "gemma_remote",
    ("bedrock-gemma-4-26b", REASONING_FIRST_PROMPT_SHA256): "gemma_bedrock_rf",
    ("remote-medpsy-4b", BASELINE_PROMPT_SHA256): "medpsy_remote",
    ("local-gemma-4-26b", REASONING_FIRST_PROMPT_SHA256): "local_gemma_mlx",
}


def profile_from_confusion(c: dict[str, int]) -> dict:
    """Derive a reader's belief parameters from its confusion counts. No tuning:
    every field is an arithmetic function of ``cc, ci, ic, ii``.

    The parameters are LIKELIHOODS — the reader's detection rates conditioned on the
    latent TRUTH (the matrix columns), which are prevalence-free reader properties,
    NOT posteriors/accuracies (those depend on the base rate). Each verdict's
    evidence weight is its log-LIKELIHOOD-RATIO; the base rate enters once, as the
    explicit prior. The reader-only Bayes calculation reproduces fit-set accuracy
    at ``prior_logodds`` for one read. Production's additional source-reliability
    floor makes the final scalar a hybrid, so this anchor must not be advertised
    as a clean deployment-prevalence knob.
    """
    cc, ci, ic, ii = c["cc"], c["ci"], c["ic"], c["ii"]
    if min(cc, ci, ic, ii) <= 0:
        raise ValueError(
            "confusion cells must all be positive to derive finite log-likelihood ratios"
        )
    n_correct = cc + ic                        # gold-correct total (matrix column)
    n_incorrect = ci + ii                      # gold-incorrect total (matrix column)
    sens = cc / n_correct                       # P(confirm | correct)   — sensitivity
    fpr = ci / n_incorrect                      # P(confirm | incorrect) — false-alarm
    return {
        "confusion": dict(c),
        # LIKELIHOODS: reader detection rates given the truth (base-rate-free)
        "sensitivity": sens,                    # P(confirm | correct)
        "false_positive_rate": fpr,             # P(confirm | incorrect)
        "specificity": 1.0 - fpr,               # P(reject | incorrect)
        "miss_rate": 1.0 - sens,                # P(reject | correct)
        # the evidence a verdict adds = its log-LIKELIHOOD-RATIO for "correct"
        "log_lr_confirm": math.log(sens / fpr),
        "log_lr_reject": math.log((1.0 - sens) / (1.0 - fpr)),
        # evidence-pair base rate in the profile fit — the default score anchor
        "prior_correct": n_correct / (n_correct + n_incorrect),
        "prior_logodds": math.log(n_correct / n_incorrect),
    }


def _named_profile(name: str) -> dict:
    profile = profile_from_confusion(_CONFUSION[name])
    profile.update(_PROFILE_META[name])
    profile.update({
        "reader_configuration": (
            f"{profile['reader_model']}@prompt-sha256:{profile['prompt_sha256']}"
        ),
        "fit_gold": "data/benchmark/eval_curation_v1.jsonl",
        "fit_gold_sha256": FIT_GOLD_SHA256,
        "fit_unique_pairs": sum(_CONFUSION[name].values()),
        "gold_rule": "exact pair; multi-curator any-incorrect-wins; duplicate pairs removed",
    })
    return profile


# Served model ids that a shipped run's call log legitimately carries even though
# they are no longer the registry's live id.  ``remote-gemma-4-26b`` was asked for
# as ``gemma-4-26b`` until 5e89e2c (2026-06-07) renamed the gateway id to
# ``gemma-4-26b-ollama``; runs exported before that rename — currently
# data/results/rasmachine_mono_gemma_remote_direct.jsonl — are honestly labelled
# and already shipped, so their historical id must keep resolving.  This table is
# provenance, not slack: only an id actually observed in a shipped run's call log
# belongs here, and it widens nothing (a name still needs its own profile).
_HISTORICAL_SERVED_MODEL_IDS: dict[str, frozenset[str]] = {
    "remote-gemma-4-26b": frozenset({"gemma-4-26b"}),
}


def _call_log_fingerprints(run_path: str | Path) -> tuple[dict[str, int], dict[str, int]]:
    """One tolerant pass over a run's call logs -> (prompt digests, served ids).

    Both fingerprints come from the same walk: run files reach 500 MB and a
    second independent pass would double the read for no new information.
    """
    prompts: Counter[str] = Counter()
    models: Counter[str] = Counter()
    try:
        with Path(run_path).open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                for call in row.get("call_log") or []:
                    # The two counters are deliberately scoped differently.
                    #
                    # PROMPT — monolithic only. A profile is keyed on the exact
                    # monolithic ``system`` string; a probe's or adjudicator's
                    # system prompt is a different artifact entirely, so counting
                    # it would manufacture a spurious 'mixed' on every decomposed
                    # run.
                    #
                    # MODEL — every call kind. The belief scalar is produced by
                    # the whole call chain, not the monolithic call alone, so a
                    # profile fitted on one homogeneous endpoint may not be
                    # claimed by a run that served part of its chain elsewhere:
                    # a heterogeneous-endpoint run is INTENTIONALLY refused as
                    # 'mixed'. Scoping this to monolithic would also blind the
                    # guard on exactly the runs with no monolithic prompt digest
                    # to verify anything else (decomposed/probe-only runs, e.g.
                    # data/results/eval_curation_v1_medpsy_decomp.jsonl, whose
                    # only served-id evidence sits on probe_* / verify_grounding
                    # calls). The asymmetry costs nothing on homogeneous runs:
                    # across data/results, 76 of 136 runs record any model_id and
                    # in 75 of them the monolithic-only id set equals the
                    # all-kinds id set.
                    system = call.get("system")
                    if call.get("kind") == "monolithic" and isinstance(system, str):
                        prompts[hashlib.sha256(system.encode("utf-8")).hexdigest()] += 1
                    model_id = call.get("model_id")
                    if isinstance(model_id, str) and model_id.strip():
                        models[model_id.strip()] += 1
    except (OSError, json.JSONDecodeError, TypeError):
        return {}, {}
    return dict(sorted(prompts.items())), dict(sorted(models.items()))


def prompt_fingerprints_for_run(run_path: str | Path) -> dict[str, int]:
    """Count monolithic system-prompt fingerprints persisted in a run.

    Rows handled without an LLM call legitimately have no call log and do not
    make a run ambiguous.  More than one digest means the run mixed scorer
    configurations and is therefore ineligible for a single calibration profile.
    """
    return _call_log_fingerprints(run_path)[0]


def model_fingerprints_for_run(run_path: str | Path) -> dict[str, int]:
    """Count served model ids persisted in a run's call logs.

    The counted value is the SERVING-layer id the endpoint was asked for (the
    registry's ``model_id``), not the canonical/registry model *name*: the same
    weights on two hosts carry different served ids, and one served id can back
    more than one registry name.  EVERY call kind counts, not just ``monolithic``
    — see the scope decision at the counting site in
    :func:`_call_log_fingerprints`.  Same tolerance as
    :func:`prompt_fingerprints_for_run` — rows handled without an LLM call, and
    older decomposed-phase call rows that recorded no ``model_id`` at all, are
    simply not observations and never make a run ambiguous.

    Residual limitation: this cannot separate configurations that differ only
    below the served id.  ``bedrock-gemma-4-26b`` and its reasoning-isolation
    twin ``bedrock-gemma-4-26b-noreason`` both serve ``google.gemma-4-26b-a4b``,
    so the served id alone does not verify reasoning mode.
    """
    return _call_log_fingerprints(run_path)[1]


def _accepted_served_model_ids(canonical: str) -> frozenset[str] | None:
    """Served ids a run under canonical model ``canonical`` may legitimately show.

    ``None`` means "no expectation on record" (unknown name, or a registry entry
    that declares no ``model_id``) — the guard then makes no claim rather than
    manufacturing a mismatch.
    """
    model_id = (LOCAL_MODELS.get(canonical) or {}).get("model_id")
    if not isinstance(model_id, str) or not model_id:
        return None
    return frozenset({model_id}) | _HISTORICAL_SERVED_MODEL_IDS.get(canonical, frozenset())


def reader_configuration_for_run(
    run_path: str | Path, model: str | None = None, *,
    prompt_sha256: str | None = None,
) -> dict:
    """Return the model+prompt identity that actually produced ``run_path``.

    Both halves of the identity are fingerprint-verified against the run's own
    call logs where the run recorded them.  A declared model that disagrees with
    the served ids on disk takes the same hard gate as a prompt disagreement:
    a mislabelled meta cannot hand an ENABLED profile to a run that a
    ship-gate-DISABLED model actually served.
    """
    path = Path(run_path)
    try:
        meta = json.loads(path.with_suffix(".meta.json").read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        meta = {}
    if not model:
        model = meta.get("model")
    declared_prompt = prompt_sha256 or meta.get("prompt_sha256")
    if not declared_prompt and isinstance(meta.get("reader_configuration"), dict):
        declared_prompt = meta["reader_configuration"].get("prompt_sha256")
    if declared_prompt:
        declared_prompt = str(declared_prompt).lower()
    canonical = canonical_model_name(model.strip().lower()) if model else None
    fingerprints, model_fingerprints = _call_log_fingerprints(path)
    observed_prompt = next(iter(fingerprints)) if len(fingerprints) == 1 else None
    if len(fingerprints) > 1:
        status = "mixed"
        resolved_prompt = None
    elif observed_prompt and declared_prompt and observed_prompt != declared_prompt:
        status = "mismatch"
        resolved_prompt = None
    else:
        resolved_prompt = observed_prompt or declared_prompt
        status = "identified" if resolved_prompt else "missing_prompt"
    # Model cross-check. Absence is never evidence: a run with no recorded served
    # id makes no claim, and neither do we. An unrecognized declared name has no
    # accepted-id expectation on record, so it can never produce a MISMATCH — but
    # it is NOT exempt from the ambiguity branch below, which fires first: a run
    # that served more than one id is refused as 'mixed' whatever it was declared
    # as, because "which endpoint produced this run" then has no answer at all.
    # A prompt mixed/mismatch already hard-gates and keeps its own, more specific
    # status — prompt precedence, pinned in tests/test_reader_configuration_model_guard.py
    # and mirrored by results._prompt_side_disagreement.
    observed_models = set(model_fingerprints)
    model_status = None
    if canonical and observed_models:
        accepted = _accepted_served_model_ids(canonical)
        if len(observed_models) > 1:
            model_status = "mixed"
        elif accepted and not observed_models <= accepted:
            model_status = "mismatch"
    if model_status and status == "identified":
        status = model_status
        # Nulling the prompt is what actually gates: calibration_for_run resolves
        # on (model, prompt_sha256) and never reads ``id``.
        resolved_prompt = None
    config_id = (
        f"{canonical}@prompt-sha256:{resolved_prompt}"
        if canonical and resolved_prompt else None
    )
    return {
        "status": status,
        "id": config_id,
        "model": canonical,
        "prompt_sha256": resolved_prompt,
        "prompt_fingerprint_source": (
            "call_log" if observed_prompt else "run_metadata" if declared_prompt else None
        ),
        "declared_prompt_sha256": declared_prompt,
        "prompt_fingerprints": fingerprints,
        "model_fingerprints": model_fingerprints,
    }


def fitted_calibration_for(
    model: str | None, *, prompt_sha256: str | None = None,
) -> dict | None:
    """Resolve any measured fit, including candidates that failed the ship gate.

    This is for diagnostics and gate reproduction. Production callers should use
    :func:`calibration_for`, which additionally enforces deployment status.
    """
    if not model or not prompt_sha256:
        return None
    canonical = canonical_model_name(model.strip().lower())
    name = _FITTED_CONFIGS.get((canonical, prompt_sha256.lower()))
    return _named_profile(name) if name else None


def calibration_for(
    model: str | None, *, prompt_sha256: str | None = None,
) -> dict | None:
    """Resolve a ship-approved exact model+prompt configuration, or ``None``.

    A model name alone is intentionally insufficient: scorer prompts can change
    while weights and serving host stay fixed.  Measured-but-failed candidates
    (currently remote MedPsy) also resolve to ``None`` in production.
    """
    profile = fitted_calibration_for(model, prompt_sha256=prompt_sha256)
    if profile is None or profile["deployment_status"] != "enabled":
        return None
    return profile


def fitted_calibration_for_run(run_path: str | Path, model: str | None = None) -> dict | None:
    """Resolve a measured diagnostic profile from a run's persisted identity."""
    config = reader_configuration_for_run(run_path, model)
    return fitted_calibration_for(config["model"], prompt_sha256=config["prompt_sha256"])


def calibration_for_run(run_path: str | Path, model: str | None = None) -> dict | None:
    """Resolve the ship-approved profile for the exact configuration in a run."""
    config = reader_configuration_for_run(run_path, model)
    return calibration_for(config["model"], prompt_sha256=config["prompt_sha256"])
