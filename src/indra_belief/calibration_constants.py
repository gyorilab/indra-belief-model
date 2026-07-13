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

from .model_client import canonical_model_name

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
}

_FITTED_CONFIGS = {
    ("remote-gemma-4-26b", BASELINE_PROMPT_SHA256): "gemma_remote",
    ("bedrock-gemma-4-26b", REASONING_FIRST_PROMPT_SHA256): "gemma_bedrock_rf",
    ("remote-medpsy-4b", BASELINE_PROMPT_SHA256): "medpsy_remote",
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


def prompt_fingerprints_for_run(run_path: str | Path) -> dict[str, int]:
    """Count monolithic system-prompt fingerprints persisted in a run.

    Rows handled without an LLM call legitimately have no call log and do not
    make a run ambiguous.  More than one digest means the run mixed scorer
    configurations and is therefore ineligible for a single calibration profile.
    """
    counts: Counter[str] = Counter()
    try:
        with Path(run_path).open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                for call in row.get("call_log") or []:
                    system = call.get("system")
                    if call.get("kind") == "monolithic" and isinstance(system, str):
                        counts[hashlib.sha256(system.encode("utf-8")).hexdigest()] += 1
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return dict(sorted(counts.items()))


def reader_configuration_for_run(
    run_path: str | Path, model: str | None = None, *,
    prompt_sha256: str | None = None,
) -> dict:
    """Return the model+prompt identity that actually produced ``run_path``."""
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
    fingerprints = prompt_fingerprints_for_run(path)
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
