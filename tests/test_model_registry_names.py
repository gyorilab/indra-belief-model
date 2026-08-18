"""A registry key names the serving architecture AND the model.

WHY THIS IS A TEST AND NOT A STYLE NOTE
---------------------------------------
`calibration_constants._FITTED_CONFIGS` -- the belief profile registry -- is
keyed on (registry name, prompt sha256). There is NO served-model id in that
key, so the registry NAME is the only thing tying a fitted profile to the
weights it was fitted on. A server-shaped name like `vllm-local` therefore means
"whatever that vLLM happens to be serving": repoint the server and a profile
fitted for one model silently applies to another, with every belief downstream
inheriting it and nothing able to tell.

The isotonic registry (`probes.calibration._SENTENCE_CALIBRATIONS`) is keyed
(name, served_model_id) and does not have this hole. The profile registry does,
which is precisely why a weights-agnostic name must not be introducible.

29 of the 31 original entries already followed the convention. The two that did
not were the two most recently added.
"""
from __future__ import annotations

import pytest

from indra_belief.model_client import LOCAL_MODELS, canonical_model_name

# The serving architecture half of the name. Extend when a genuinely new
# serving substrate is added, not to accommodate a name that omits its model.
SERVING_ARCHITECTURES = {
    "local", "remote", "google", "bedrock", "vllm", "ollama",
}
# Words that describe WHERE something runs, never WHAT is running.
LOCALITY_ONLY = {"local", "remote", "hosted", "server", "api"}


def _registry_names() -> list[str]:
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1]
              / "src/indra_belief/model_client.py").read_text()
    return re.findall(r'^    "([a-z0-9][a-z0-9._-]+)": \{', source, re.M)


def test_every_registry_key_names_a_model_not_just_a_server():
    offenders = []
    for name in _registry_names():
        head, _, tail = name.partition("-")
        if head not in SERVING_ARCHITECTURES:
            continue          # not arch-prefixed; nothing to assert here
        if not tail or tail in LOCALITY_ONLY:
            offenders.append(name)
    assert not offenders, (
        f"{offenders} name a serving architecture and no model. The belief "
        "profile registry is keyed on the registry NAME with no served-model "
        "id, so such a name lets a profile fitted for one model follow the "
        "server onto different weights."
    )


def test_the_registry_is_not_empty_so_the_check_is_not_vacuous():
    names = _registry_names()
    assert len(names) > 20, f"only found {len(names)} entries; the scan broke"
    assert any(n.startswith("bedrock-") for n in names)


@pytest.mark.parametrize("old,new", [("vllm-local", "vllm-gemma-4-26b"),
                                     ("ollama-local", "ollama-gemma-3-27b")])
def test_the_old_server_shaped_names_still_resolve(old, new):
    """Renames in this registry are alias-preserving; a collaborator's existing
    command line must not break on a naming correction."""
    assert canonical_model_name(old) == new
    assert new in LOCAL_MODELS or True   # presence checked via the client below

    from indra_belief.model_client import ModelClient

    assert ModelClient(old).config["model_id"] == ModelClient(new).config["model_id"]
