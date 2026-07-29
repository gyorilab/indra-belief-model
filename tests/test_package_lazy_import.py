from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_transport_submodule_does_not_pull_indra_or_scorer_closure() -> None:
    source = r'''
import json
import sys
sys.path.insert(0, SOURCE_ROOT)
import indra_belief.model_client
import indra_belief.spend_guard
print(json.dumps(sorted(
    name for name in sys.modules
    if name == "indra" or name.startswith("indra.")
    or name.startswith("indra_belief.scorers")
)))
'''.replace("SOURCE_ROOT", repr(str(ROOT / "src")))
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-P", "-c", source],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == []


def test_public_convenience_api_remains_available() -> None:
    import indra_belief

    assert indra_belief.ModelClient.__name__ == "ModelClient"
    assert indra_belief.ModelResponse.__name__ == "ModelResponse"
    assert callable(indra_belief.score_evidence)
    assert callable(indra_belief.score_statement)

