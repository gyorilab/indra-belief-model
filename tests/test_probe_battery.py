"""Contract tests for the no-reasoning battery using synthetic records only."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import textwrap
from collections import Counter
from collections.abc import Iterator, Mapping

import pytest

from indra_belief.probes.battery import (
    LABELS,
    LABEL_TOKEN_IDS,
    PROBES,
    PROBE_FAMILIES,
    PROBE_IDS,
    RENDER_FIELDS,
    TAXONOMY_TAGS,
    Probe,
    battery_digest,
    oriented_p,
    probe_by_id,
    probes_in_family,
    render,
    required_fields,
)


EXPECTED_PROBE_SHAPE = (
    ("tax.relation_present", "taxonomy", "direct", ("no_relation",)),
    (
        "tax.subject_grounded",
        "taxonomy",
        "direct",
        ("grounding", "entity_boundaries"),
    ),
    (
        "tax.object_grounded",
        "taxonomy",
        "direct",
        ("grounding", "entity_boundaries"),
    ),
    ("tax.relation_type", "taxonomy", "direct", ("wrong_relation",)),
    ("tax.activity_vs_amount", "taxonomy", "direct", ("act_vs_amt",)),
    ("tax.direction_polarity", "taxonomy", "direct", ("polarity",)),
    (
        "tax.assertion_not_hypothesis",
        "taxonomy",
        "direct",
        ("hypothesis",),
    ),
    (
        "tax.not_negative_result",
        "taxonomy",
        "direct",
        ("negative_result",),
    ),
    ("tax.mod_site_match", "taxonomy", "direct", ("mod_site",)),
    ("pol.verdict_direct", "polarity", "direct", ()),
    ("pol.verdict_flipped", "polarity", "flipped", ()),
    ("pol.relation_direct", "polarity", "direct", ("no_relation",)),
    ("pol.relation_flipped", "polarity", "flipped", ("no_relation",)),
    ("perturb.paraphrase", "perturbation", "direct", ()),
    ("perturb.evidence_first", "perturbation", "direct", ()),
    ("perturb.field_order", "perturbation", "direct", ()),
)

SYNTHETIC_RECORD = {
    "subject": "PROTX",
    "object": "PROTY",
    "stmt_type": "Activation",
    "evidence_text": "Synthetic PROTX activates PROTY while GENEZ is unchanged.",
}


def test_probe_shape_counts_and_frozen_id_order():
    shape = tuple(
        (probe.id, probe.family, probe.orientation, probe.targets)
        for probe in PROBES
    )
    assert shape == EXPECTED_PROBE_SHAPE
    assert len(PROBES) == 16
    assert Counter(probe.family for probe in PROBES) == {
        "taxonomy": 9,
        "polarity": 4,
        "perturbation": 3,
    }
    assert PROBE_IDS == tuple(probe.id for probe in PROBES)
    assert len(PROBE_IDS) == len(set(PROBE_IDS))


def test_probe_questions_are_pairwise_distinct():
    pairs = tuple((probe.system, probe.user_template) for probe in PROBES)
    assert len(pairs) == len(set(pairs))


def test_declared_vocabularies_and_taxonomy_coverage():
    assert PROBE_FAMILIES == ("taxonomy", "polarity", "perturbation")
    assert TAXONOMY_TAGS == (
        "correct",
        "no_relation",
        "grounding",
        "wrong_relation",
        "other",
        "act_vs_amt",
        "entity_boundaries",
        "hypothesis",
        "polarity",
        "negative_result",
        "mod_site",
        "agent_conditions",
    )
    for probe in PROBES:
        assert probe.family in PROBE_FAMILIES
        assert probe.orientation in {"direct", "flipped"}
        assert set(probe.targets) <= set(TAXONOMY_TAGS)

    taxonomy_targets = {
        target
        for probe in PROBES
        if probe.family == "taxonomy"
        for target in probe.targets
    }
    assert taxonomy_targets == {
        "no_relation",
        "grounding",
        "wrong_relation",
        "act_vs_amt",
        "entity_boundaries",
        "hypothesis",
        "polarity",
        "negative_result",
        "mod_site",
    }


def test_templates_use_only_the_content_render_surface():
    """Use three shared content fields plus TEST evidence recovered by join."""
    assert RENDER_FIELDS == frozenset(
        {"subject", "object", "stmt_type", "evidence_text"}
    )
    assert all(required_fields(probe) <= RENDER_FIELDS for probe in PROBES)

    local_probe = Probe(
        id="synthetic.parser_check",
        family="taxonomy",
        system="Synthetic contract.",
        user_template="{subject} / {evidence_text} / {subject}",
        prefill_suffix='{"verdict":"',
        orientation="direct",
        targets=(),
    )
    assert required_fields(local_probe) == frozenset({"subject", "evidence_text"})


def test_prompts_are_short_closed_and_have_no_reasoning_scaffolding():
    for probe in PROBES:
        assert probe.system.strip()
        assert len(probe.system) <= 800
        assert probe.user_template.strip()
        prompt = f"{probe.system}\n{probe.user_template}".lower()
        assert "example" not in prompt
        assert "explain" not in prompt
        assert "justify" not in prompt
        assert "quote" not in prompt
        assert "confidence" not in prompt


def test_prefills_stop_at_the_unlabelled_value_delimiter():
    for probe in PROBES:
        assert probe.prefill_suffix
        assert "correct" not in probe.prefill_suffix.lower()
        assert "incorrect" not in probe.prefill_suffix.lower()
        assert probe.prefill_suffix.endswith(('"', ": "))


class _TrackingRecord(Mapping[str, object]):
    def __init__(self, values: Mapping[str, object]) -> None:
        self._values = dict(values)
        self.reads: list[str] = []

    def __getitem__(self, key: str) -> object:
        self.reads.append(key)
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


@pytest.mark.parametrize("probe", PROBES, ids=PROBE_IDS)
def test_render_round_trip_and_reads_only_required_fields(probe: Probe):
    record = _TrackingRecord({**SYNTHETIC_RECORD, "unused": "GENEZ"})
    system, user, prefill = render(probe, record)

    assert system == probe.system
    assert prefill == probe.prefill_suffix
    assert system and user and prefill
    assert "PROTX" in user
    assert "PROTY" in user
    assert SYNTHETIC_RECORD["evidence_text"] in user
    assert record.reads == sorted(required_fields(probe))


def test_render_key_error_names_the_missing_field():
    probe = probe_by_id("tax.relation_present")
    record = dict(SYNTHETIC_RECORD)
    del record["evidence_text"]

    with pytest.raises(KeyError, match="evidence_text"):
        render(probe, record)


def test_probe_lookup_and_family_filter_fail_closed():
    assert probe_by_id(PROBE_IDS[0]) is PROBES[0]
    assert probes_in_family("polarity") == PROBES[9:13]

    with pytest.raises(KeyError, match="unknown probe id"):
        probe_by_id("missing.probe")
    with pytest.raises(ValueError, match="unknown probe family"):
        probes_in_family("missing")


def test_oriented_probability_contract():
    assert oriented_p(0.8, "direct") == 0.8
    assert oriented_p(0.8, "flipped") == pytest.approx(0.2)
    assert oriented_p(0.5, "flipped") == 0.5
    with pytest.raises(ValueError, match="orientation"):
        oriented_p(0.8, "sideways")


def _local_battery_digest(probes: tuple[Probe, ...]) -> str:
    payload = {
        "labels": LABELS,
        "label_token_ids": LABEL_TOKEN_IDS,
        "probes": [
            (
                probe.id,
                probe.family,
                probe.system,
                probe.user_template,
                probe.prefill_suffix,
                probe.orientation,
                probe.targets,
            )
            for probe in probes
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_battery_digest_is_stable_hex_and_order_sensitive():
    first = battery_digest()
    second = battery_digest()
    assert first == second
    assert re.fullmatch(r"[0-9a-f]{64}", first)
    assert first == _local_battery_digest(PROBES)

    permuted = PROBES[1:] + PROBES[:1]
    assert _local_battery_digest(permuted) != first


def test_closed_labels_and_verified_token_ids():
    assert LABELS == ("correct", "incorrect")
    assert LABEL_TOKEN_IDS == (19448, 111863)


def test_import_does_not_pull_heavy_runner_dependencies():
    """Keep the shared declaration independent across its two environments.

    The combiner/evaluator consumes it in ``.venv`` while the MLX runner uses
    ``~/.venvs/mlx-serve``; loading either track's heavy stack here would couple
    the two.
    """
    program = textwrap.dedent(
        """
        import sys
        import indra_belief.probes.battery

        heavy_roots = {"mlx", "mlx_lm", "torch", "numpy", "sklearn", "transformers"}
        loaded = sorted(
            name for name in sys.modules if name.split(".", 1)[0] in heavy_roots
        )
        if loaded:
            raise SystemExit("heavy imports: " + ", ".join(loaded))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
