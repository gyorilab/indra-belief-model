"""Gate the viewer's paper-literal calibration MLE against the Python canonical.

Shells the contract test (which asserts the TS `calibrationInterceptSlope` port
reproduces `metrics.py:_calibration_intercept_slope` to ~1e-6, plus the pure data
contract) through Node's native type-stripping. Skipped if node is unavailable.

Also gates the AP-decomposition artifact the /paper figure draws: each arm's band
nets must sum to that arm's point ΔAP to 1e-9, and the whiskers must be the
shipped paired-bootstrap bounds unit-converted and nothing else. The .mjs asserts
the same properties on the TS side; this is the Python end of that parity, so the
figure cannot drift from the shipped number in either language. The BANDING is
gated too, and independently: the bands are cut on an exogenous evidence census,
and the per-band populations are re-derived here from the shared gold's own
``evidence_review.corpus_evidence_entries`` rather than read off the artifact.

The same treatment covers the review-queue artifact behind /paper beat 2: the bar
geometry is ``queue == true_errors_caught + false_alarms`` with
``precision == caught / queue``, so both identities are asserted here and in the
.mjs, along with the run manifest's sha256 for the exact bytes drawn.

Two more artifacts from the same run are gated the same way. The belief-model
ladder: every entry's pooled average precision is RE-DERIVED here from the
prediction file the entry names, and every delta must be that value minus the
baseline's. The non-reading control: the raw row must be the shipped SimpleScorer
pooled average precision and the full control must sit strictly below it, with an
internally consistent route census and de-dup arithmetic. Re-derivation uses the
small prediction files and the artifacts' own recorded counts — the 19 MB
execution map is the compute script's own assertion to own, not pytest's.

The framing correction gets the same treatment, and its headline gets more: leg
(b) — no reader belief exceeds the noisy-OR on the same statement — is RE-DERIVED
here from the four 1689-row prediction files against the shipped SimpleScorer
scores, not read off the artifact, because that single claim is what licenses
calling the reader arm the paper's own aggregation. Leg (a) is asserted against
what the four bundle manifests actually say AND against the sha256 of the
aggregation config and the two implementation sources in the tree. Leg (c)'s
reachable-set enumeration is NOT re-run here — same precedent as the execution
map: the compute script exits non-zero on a counterexample and owns that check.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from indra_belief.comparison import metrics


ROOT = Path(__file__).resolve().parents[1]
TS_RUNNER = ROOT / "viewer" / "scripts" / "test-paper-literal-contract.mjs"
# The cross-cutting sweep: the three defect classes that have each shipped more
# than once across the whole /paper surface (a frozen join key rendered as a
# display name, an unenforced SVG label budget, a placeholder that renders as a
# measurement). Per-figure runners are exactly what missed them.
TS_RENDER_INVARIANTS = ROOT / "viewer" / "scripts" / "test-paper-render-invariants.mjs"

_MODEL_DIR = ROOT / "data" / "results" / "indra_paper_literal_models_20260724"
_DECOMPOSITION_PATH = _MODEL_DIR / "ap_decomposition_by_paper_band.json"
_REVIEW_QUEUE_PATH = _MODEL_DIR / "statement_review_queue.json"
_LADDER_PATH = _MODEL_DIR / "belief_model_ladder.json"
_NON_READING_CONTROL_PATH = _MODEL_DIR / "non_reading_control.json"
_VS_LLMS_PATH = _MODEL_DIR / "paper_literal_vs_llms.json"
_RUN_MANIFEST_PATH = _MODEL_DIR / "manifest.json"

# The figure's fixed, un-broken y-domain in AP points (viewer/src/lib/data/
# paper-ap-decomposition.ts). Widening it is a regression, so pin it from here too.
_Y_DOMAIN_PTS = (-2.8, 2.1)
# Band nets sum to the point ΔAP this tightly; the shipped `delta` field is a
# BOOTSTRAP MEAN, so it is only held to 1e-4 (the figure never draws it).
_PARITY_TOL = 1e-9
_BOOTSTRAP_MEAN_TOL = 1e-4
# The frozen evidence-count ladder the figure's x-axis IS (viewer/src/lib/data/
# paper-ap-decomposition.ts::AP_DECOMP_BAND_EDGES). ``None`` = open upper end.
_BAND_EDGES = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, None)]


def _load(path: Path) -> dict:
    return json.loads(path.read_text())

# The TS golden fixture (viewer/scripts/test-paper-literal-contract.mjs) pins the
# TS port to these two literals. Anchor the PYTHON side to the SAME literals so
# the cross-language parity is enforced on both ends: without this, a mutation to
# metrics._calibration_intercept_slope would drift silently (the .mjs never runs
# Python). Fixture: well-conditioned scores + mixed-class labels, all weights 1.
_CALIB_SCORES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], float)
_CALIB_LABELS = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 1], bool)
_CALIB_GOLDEN_INTERCEPT = 0.6500881338450821
_CALIB_GOLDEN_SLOPE = 2.9968037431899788


def test_python_calibration_mle_matches_cross_language_golden() -> None:
    # Python end of the TS<->Python calibration golden (the .mjs pins the TS end).
    intercept, slope = metrics._calibration_intercept_slope(
        _CALIB_LABELS, _CALIB_SCORES, np.ones(len(_CALIB_SCORES)), epsilon=1e-6
    )
    assert intercept == pytest.approx(_CALIB_GOLDEN_INTERCEPT, abs=1e-9)
    assert slope == pytest.approx(_CALIB_GOLDEN_SLOPE, abs=1e-9)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_viewer_paper_literal_contract() -> None:
    completed = subprocess.run(
        ["node", "--experimental-strip-types", str(TS_RUNNER)],
        cwd=ROOT / "viewer",
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, (
        "viewer paper-literal calibration parity assertions failed:\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_viewer_paper_render_invariants() -> None:
    completed = subprocess.run(
        ["node", "--experimental-strip-types", str(TS_RENDER_INVARIANTS)],
        cwd=ROOT / "viewer",
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, (
        "/paper render-invariant assertions failed:\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


# Mirrors BELIEF_LADDER_DISPLAY_BUDGET_CHARS. The .mjs re-derives it from the
# geometry; this side pins the same number so the two cannot drift apart silently.
_LADDER_DISPLAY_MAX_CHARS = 39


def test_ladder_display_names_fit_the_axis():
    """The rendered rung names, not the join keys, are what can clip.

    `label` addresses shipped bytes and is never drawn; `display` is. Both are
    checked, because a budget on the wrong one of the two is how this class of
    defect survived the first fix.
    """
    from pathlib import Path as _Path
    import re as _re

    source = (
        ROOT / "viewer" / "src" / "lib" / "data" / "paper-belief-ladder.ts"
    ).read_text()
    displays = _re.findall(r"display: '([^']+)',\n\t\tkind: '", source)
    displays += _re.findall(r"display: '([^']+)', kind: '", source)
    assert len(displays) == len(_LADDER_ENTRIES), (
        f"expected one display per rung, found {len(displays)}"
    )
    for name in displays:
        assert len(name) <= _LADDER_DISPLAY_MAX_CHARS, (
            f"{name!r} is {len(name)} chars; the axis clips past "
            f"{_LADDER_DISPLAY_MAX_CHARS}"
        )


def test_ap_decomposition_band_nets_sum_to_point_delta() -> None:
    """The figure's endpoint IS the arm's ΔAP — reached, not asserted."""
    decomposition = _load(_DECOMPOSITION_PATH)
    arms = decomposition["arms"]
    assert len(arms) == 5, "the figure draws exactly five arms"

    for arm in arms:
        nets = [float(v) for v in arm["per_band_net_pts"]]
        cumulative = [float(v) for v in arm["cumulative_pts"]]
        assert len(nets) == len(_BAND_EDGES), arm["name"]
        assert len(cumulative) == len(_BAND_EDGES), arm["name"]

        # (1) the band nets sum to the arm's point ΔAP
        assert math.fsum(nets) == pytest.approx(arm["total_pts"], abs=_PARITY_TOL), arm["name"]
        # (2) the drawn cumulative series is exactly their running sum
        running = 0.0
        for index, net in enumerate(nets):
            running += net
            assert running == pytest.approx(cumulative[index], abs=_PARITY_TOL), (
                f"{arm['name']} band {index + 1}"
            )
        assert cumulative[-1] == pytest.approx(arm["total_pts"], abs=_PARITY_TOL), arm["name"]
        # (3) points and AP agree (1 pt = 0.01 AP)
        assert arm["total_pts"] == pytest.approx(
            arm["total_delta_ap"] * 100.0, abs=_PARITY_TOL
        ), arm["name"]
        # (4) nothing drawn escapes the fixed, un-broken y-domain
        low, high = _Y_DOMAIN_PTS
        for value in [*cumulative, arm["ci95_low_pts"], arm["ci95_high_pts"]]:
            assert low <= value <= high, f"{arm['name']} escapes the fixed y-domain"


def test_ap_decomposition_matches_shipped_head_to_head() -> None:
    """Whiskers and shares are the shipped bounds; the dot is the POINT delta."""
    decomposition = _load(_DECOMPOSITION_PATH)
    vs_llms = _load(_VS_LLMS_PATH)
    paired = vs_llms["paired_delta_vs_paper_literal"]

    assert decomposition["n_statements"] == vs_llms["n_statements"]
    assert decomposition["provenance"]["bootstrap"]["n_bootstrap"] == vs_llms["n_bootstrap"]
    assert decomposition["provenance"]["bootstrap"]["seed"] == vs_llms["seed"]
    assert (
        decomposition["reference_average_precision"]
        == vs_llms["point_metrics"]["Paper literal RF+promoter"]["pooled_average_precision"]
    )

    for arm in decomposition["arms"]:
        shipped = paired[arm["name"]]["pooled_average_precision"]
        # The point delta and the shipped bootstrap MEAN are different statistics;
        # they agree closely, and the figure draws the former.
        assert arm["total_delta_ap"] == pytest.approx(
            shipped["delta"], abs=_BOOTSTRAP_MEAN_TOL
        ), arm["name"]
        assert arm["ci95_low_pts"] == shipped["ci95_low"] * 100.0, arm["name"]
        assert arm["ci95_high_pts"] == shipped["ci95_high"] * 100.0, arm["name"]
        assert arm["p_arm_greater"] == shipped["p_arm_greater"], arm["name"]
        assert arm["clears_zero"] == (
            shipped["ci95_low"] > 0 or shipped["ci95_high"] < 0
        ), arm["name"]


def test_ap_decomposition_is_the_artifact_the_manifest_records() -> None:
    """The viewer draws the exact bytes the run manifest signed."""
    manifest = _load(_RUN_MANIFEST_PATH)
    recorded = manifest["output_sha256"][_DECOMPOSITION_PATH.name]
    digest = hashlib.sha256(_DECOMPOSITION_PATH.read_bytes()).hexdigest()
    assert digest == recorded


def test_ap_decomposition_band_counts_are_internally_consistent() -> None:
    """The printed count strip is the artifact's own banding, not a caption."""
    decomposition = _load(_DECOMPOSITION_PATH)
    bands = decomposition["bands"]
    assert [band["n_true"] for band in bands] == decomposition["band_true_counts"]
    assert [band["n_false"] for band in bands] == decomposition["band_false_counts"]
    assert sum(decomposition["band_true_counts"]) == decomposition["n_true"]
    assert sum(decomposition["band_false_counts"]) == decomposition["n_false"]
    assert (
        decomposition["n_true"] + decomposition["n_false"] == decomposition["n_statements"]
    )

    # Band membership is a pure function of the evidence count, so nothing is
    # assigned by a tie-break. That is the property the earlier decile banding had
    # to argue for one boundary at a time; here it holds by construction and the
    # artifact must still say so.
    checks = decomposition["checks"]
    assert checks["n_statements_assigned_by_a_tie_break"] == 0
    assert checks["band_membership_is_a_function_of_the_banding_variable_alone"] is True
    assert checks["bands_partition_the_panel"] is True


def test_ap_decomposition_banding_is_the_exogenous_evidence_census() -> None:
    """The bands are re-derived here, not read off the artifact.

    The banding variable is the whole point of this figure: the previous version
    banded on the reference arm's own out-of-fold score, which conditions on that
    score's estimation noise and manufactures a shape that reverses when you band
    on the compared arm instead. So the census is re-derived from the shared gold's
    own ``evidence_review.corpus_evidence_entries``, the ladder is re-applied, and
    the resulting populations must equal the artifact's band strip exactly.
    """
    decomposition = _load(_DECOMPOSITION_PATH)
    banding = decomposition["banding"]
    assert banding["kind"] == "power_of_two_ladder_on_evidence_count"
    assert banding["variable_is_exogenous"] is True
    assert [tuple(edge) for edge in banding["edges"]] == _BAND_EDGES

    evidence: dict[str, int] = {}
    labels: dict[str, int] = {}
    for line in _GOLD_PATH.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row["canonical_corpus"]["statement_id"]
        evidence[sid] = int(row["evidence_review"]["corpus_evidence_entries"])
        labels[sid] = int(row["paper_replication_policy"]["released_paper_correct"])
        # The paper's OWN released per-source counts are the same census.
        assert evidence[sid] == sum(row["paper_eligibility"]["historical_all_source_counts"]), sid

    sids, y, _mhash = _panel()
    assert set(sids) <= set(evidence)
    assert [labels[sid] for sid in sids] == [int(v) for v in y]

    for index, (low, high) in enumerate(_BAND_EDGES):
        members = [
            sid for sid in sids if evidence[sid] >= low and (high is None or evidence[sid] <= high)
        ]
        band = decomposition["bands"][index]
        assert band["n"] == len(members), band["label"]
        assert band["n_true"] == sum(labels[sid] for sid in members), band["label"]
        assert band["n_false"] == sum(1 - labels[sid] for sid in members), band["label"]
        assert band["evidence_entries"] == sum(evidence[sid] for sid in members), band["label"]

    # Every statement lands in exactly one band, so the strip is a partition.
    assert sum(band["n"] for band in decomposition["bands"]) == len(sids)


# The six arms the review-queue figure draws, in the artifact's fixed presentation
# order (mirrors REVIEW_QUEUE_ARM_SPECS in viewer/src/lib/data/paper-review-queue.ts).
_REVIEW_QUEUE_ARMS = [
    ('INDRA SimpleScorer, default priors', 'paper-model'),
    ('INDRA SimpleScorer + hierarchy', 'paper-model'),
    ('Paper RF 2k-d13 + Type/#PMIDs/promoter', 'paper-model'),
    ('Gemma 4 26B gate', 'llm-gate'),
    ('GLM-5 gate', 'llm-gate'),
    ('Gemma 4 31B gate', 'llm-gate'),
    ('Gemma 4 E2B gate', 'llm-gate'),
    ('Our BayesianScorer, source+subtype refit', 'paper-model'),
    ('Our CountsScorer RF, full features', 'paper-model'),
]

# Same geometry trap as the ladder: ReviewQueue right-anchors these at
# GUTTER_RIGHT - 4 = 206 inside viewBox="0 0 900 340"; --mono at 9px measures
# 5.4186 units/char, so 206 / 5.4186 = 38 chars before the viewport clips the
# leading glyphs. <desc> emits the full string, so only this assertion can see it.
_REVIEW_QUEUE_DISPLAY_MAX_CHARS = 38


def test_ap_decomposition_traces_are_individually_distinguishable():
    """Every fan slot needs its OWN (stroke, dash) pair.

    The original spec gave the three top LLM arms one shared hue on the reasoning
    that they trace the same path, and let identity resolve at the fan ~1000 user
    units to the right. On screen that left three overlapping olive lines with no
    way to tell them apart. Redundant encoding (luminance ramp + dash signature)
    also survives greyscale printing and colour-vision deficiency, which hue alone
    does not -- so a future edit collapsing these back to one hue is a regression,
    not a simplification.
    """
    from pathlib import Path
    import re

    src = Path("viewer/src/lib/data/paper-ap-decomposition.ts").read_text()
    body = src[src.index("AP_DECOMP_FAN_SLOTS"):]
    slots = re.findall(
        r"label: '([^']+)'.*?display: '([^']+)',\s*stroke: '([^']+)'(?:,\s*dash: '([^']+)')?",
        body,
        re.S,
    )
    assert len(slots) == 5, f"expected 5 fan slots, parsed {len(slots)}"

    signatures = [(stroke, dash or "solid") for _l, _d, stroke, dash in slots]
    assert len(set(signatures)) == len(signatures), (
        f"fan traces must be individually identifiable; got duplicates in {signatures}"
    )
    # display must never be the frozen point_metrics join key
    for label, display, _stroke, _dash in slots:
        if label.startswith("Paper literal"):
            assert display != label, "display must not render the frozen join key"



def test_review_queue_display_names_fit_the_gutter():
    from pathlib import Path
    import re

    spec_src = Path("viewer/src/lib/data/paper-review-queue.ts").read_text()
    displays = re.findall(r"display: '([^']+)'", spec_src)
    assert len(displays) == len(_REVIEW_QUEUE_ARMS), "every arm needs a display name"
    for name in displays:
        assert len(name) <= _REVIEW_QUEUE_DISPLAY_MAX_CHARS, (
            f"{name!r} is {len(name)} chars; the gutter clips past "
            f"{_REVIEW_QUEUE_DISPLAY_MAX_CHARS}"
        )

_REVIEW_QUEUE_CAVEAT_COUNT = 7


def _review_queue_points(arm: dict) -> list[dict]:
    return [arm["operating_point"], *arm["precision_at_matched_recall"]]


def test_review_queue_bar_segments_sum_to_the_bar() -> None:
    """The figure's two segments ARE the bar length, and the label IS caught/queue."""
    queue = _load(_REVIEW_QUEUE_PATH)
    panel = queue["panel"]
    assert panel["n_errors"] + panel["n_correct"] == panel["n"]
    assert panel["error_base_rate"] == pytest.approx(
        panel["n_errors"] / panel["n"], abs=_PARITY_TOL
    )

    arms = queue["arms"]
    assert [(arm["name"], arm["kind"]) for arm in arms] == _REVIEW_QUEUE_ARMS

    for arm in arms:
        for point in _review_queue_points(arm):
            where = f"{arm['name']} @ {point['target_recall']}"
            # (1) queue == real + wasted
            assert (
                point["true_errors_caught"] + point["false_alarms"] == point["queue"]
            ), where
            # (2) precision == real / queue
            assert point["precision"] == pytest.approx(
                point["true_errors_caught"] / point["queue"], abs=_PARITY_TOL
            ), where
            # (3) the achieved recall is caught/errors and never undershoots its target
            assert point["recall_achieved"] == pytest.approx(
                point["true_errors_caught"] / panel["n_errors"], abs=_PARITY_TOL
            ), where
            assert point["recall_achieved"] + _PARITY_TOL >= point["target_recall"], where
            assert point["queue"] <= panel["n"], where

        # (4) the drawn bar is the grid cell the coarseness note cites
        headline = [
            cell
            for cell in arm["precision_at_matched_recall"]
            if cell["target_recall"] == queue["headline_target_recall"]
        ]
        assert len(headline) == 1, arm["name"]
        # The grid cell carries two extra presentation fields (recall_overshoot,
        # queue_share_of_panel); every field the bar draws must be identical.
        operating = arm["operating_point"]
        assert {key: headline[0][key] for key in operating} == operating, arm["name"]
        assert arm["n_distinct_queue_sizes_across_targets"] == len(
            {cell["queue"] for cell in arm["precision_at_matched_recall"]}
        ), arm["name"]


def test_review_queue_zero_pile_is_an_llm_gate_property() -> None:
    """belief == 0 is a reader decision; a paper belief model never makes it."""
    queue = _load(_REVIEW_QUEUE_PATH)
    n_errors = queue["panel"]["n_errors"]

    for arm in queue["arms"]:
        pile = arm["zero_pile"]
        if arm["kind"] == "paper-model":
            assert pile is None, arm["name"]
            continue
        assert pile is not None, arm["name"]
        assert pile["true_errors"] + pile["false_alarms"] == pile["size"], arm["name"]
        assert pile["precision"] == pytest.approx(
            pile["true_errors"] / pile["size"], abs=_PARITY_TOL
        ), arm["name"]
        assert pile["share_of_all_errors"] == pytest.approx(
            pile["true_errors"] / n_errors, abs=_PARITY_TOL
        ), arm["name"]
        if pile["is_whole_flag_set_at_headline_target"]:
            operating = arm["operating_point"]
            assert pile["size"] == operating["queue"], arm["name"]
            assert pile["true_errors"] == operating["true_errors_caught"], arm["name"]
            assert pile["false_alarms"] == operating["false_alarms"], arm["name"]

    # The zero-pile identity holds only under the unfitted hard gate: under a
    # fitted soft profile belief floors at sigmoid(prior), not 0, and the callout
    # would silently mean something else.
    assert queue["checks"]["llm_bundles_use_unfitted_hard_gate"] == "indra_default_hard_gate"
    assert queue["checks"]["llm_zero_pile_is_the_minimum_score_tie_block"] is True


def test_review_queue_scores_the_head_to_head_panel_and_keeps_its_caveats() -> None:
    """Same 1689-statement panel as the AP table, and all method caveats travel."""
    queue = _load(_REVIEW_QUEUE_PATH)
    vs_llms = _load(_VS_LLMS_PATH)
    assert queue["panel"]["n"] == vs_llms["n_statements"]
    assert queue["headline_target_recall"] in queue["target_recalls"]
    assert len(queue["caveats"]) == _REVIEW_QUEUE_CAVEAT_COUNT
    assert all(isinstance(c, str) and c.strip() for c in queue["caveats"])


def test_review_queue_promotion_ceiling_rederives_from_the_gold_and_scores() -> None:
    """The ceiling count is CHECKED, not transcribed.

    /paper prints this number as what the gate design costs before any reader
    runs, so re-derive it here straight from the two source files — the gold
    labels and the unfiltered noisy-OR predictions — without touching the
    compute script's own loader. Also re-assert the premise the ceiling rests
    on: a reader arm can only remove belief, so none of these statements can be
    lifted back over the bar.
    """
    ceiling = _load(_REVIEW_QUEUE_PATH)["promotion_ceiling"]
    threshold = float(ceiling["threshold"])
    assert 0.0 < threshold < 1.0

    labels: dict[str, int] = {}
    for line in _GOLD_PATH.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        policy = row.get("paper_replication_policy") or {}
        if policy.get("released_paper_correct") is None:
            continue
        labels[row["canonical_corpus"]["statement_id"]] = int(
            policy["released_paper_correct"]
        )

    noisy_or = _jsonl_scores(_SIMPLE_SCORER_PREDICTIONS)
    assert set(noisy_or) == set(labels)

    true_sids = [sid for sid, y in labels.items() if y == 1]
    below = [sid for sid in true_sids if noisy_or[sid] < threshold]

    assert ceiling["n_true"] == len(true_sids)
    assert ceiling["n_true_below_threshold"] == len(below)
    assert ceiling["reference_arm"] == "INDRA SimpleScorer, default priors"
    # The subtractive premise, on the statements the claim is about.
    for arm in _FRAMING_ARMS:
        scores = _jsonl_scores(_MODELS_DIR / arm / "all_source_predictions.jsonl")
        assert all(scores[sid] <= noisy_or[sid] for sid in below), arm

    # And the panel identity the viewer validator re-checks on its own side.
    panel = _load(_REVIEW_QUEUE_PATH)["panel"]
    assert ceiling["n_true"] == panel["n_correct"]
    assert ceiling["n_true_below_threshold"] <= panel["n_correct"]


def test_review_queue_is_the_artifact_the_manifest_records() -> None:
    """The viewer draws the exact bytes the run manifest signed."""
    manifest = _load(_RUN_MANIFEST_PATH)
    assert manifest["outputs"]["review_queue"] == _REVIEW_QUEUE_PATH.name
    recorded = manifest["output_sha256"][_REVIEW_QUEUE_PATH.name]
    digest = hashlib.sha256(_REVIEW_QUEUE_PATH.read_bytes()).hexdigest()
    assert digest == recorded


# --- belief-model ladder + non-reading control (scripts/compute_belief_model_ladder.py,
# --- scripts/compute_non_reading_control.py) --------------------------------------

# The ladder's fixed presentation order (LADDER in the compute script): the
# paper's own family ascending, then the paper's literal released model, then the
# reading gates. Pinned here so a reordering is a visible change, not a silent one.
_LADDER_ENTRIES = [
    ("CountsScorer RF, source counts", "paper-family"),
    ("Hierarchy propagation", "paper-family"),
    ("noisy-OR SimpleScorer (direct)", "paper-family"),
    ("BayesianScorer, source refit", "paper-family"),
    ("BayesianScorer, source+subtype refit", "paper-family"),
    ("CountsScorer RF, full features", "paper-family"),
    ("HybridScorer, full features", "paper-family"),
    ("RF 2k-d13 + Type/#PMIDs/promoter", "paper-literal"),
    ("Gemma 4 26B gate", "reader-gate"),
    ("GLM-5 gate", "reader-gate"),
    ("Gemma 4 31B gate", "reader-gate"),
    ("Gemma 4 E2B gate", "reader-gate"),
]
_LADDER_BASELINE = "noisy-OR SimpleScorer (direct)"
# Axis geometry budget, mirrored in viewer/scripts/test-paper-literal-contract.mjs.
_LADDER_LABEL_MAX_CHARS = 39


def test_ladder_labels_fit_the_axis():
    """A label wider than the axis loses its leading glyph to the SVG viewport.

    BeliefModelLadder.svelte draws these right-anchored at x = LABEL_RIGHT - 2 =
    228 inside viewBox="0 0 900 518"; --mono at 9px measures 5.4186 units/char,
    so 228 / 5.4186 = 42 chars hard and 39 once the figure's 12-unit left gutter
    is respected. <desc> emits the full string either way, so a11y checks cannot
    see the clip -- only this assertion can.
    """
    for label, _kind in _LADDER_ENTRIES:
        assert len(label) <= _LADDER_LABEL_MAX_CHARS, (
            f"{label!r} is {len(label)} chars; the axis clips past "
            f"{_LADDER_LABEL_MAX_CHARS}"
        )

# The correct INDRA aggregation. The other form, 1 - PROD (1-r_s)^n, is wrong and
# was purged from two headers; it must not reappear in either artifact.
_NOISY_OR_FORMULA = "belief = 1 - PROD_s (syst_s + rand_s^{n_s})"
_WRONG_NOISY_OR_FRAGMENT = "1 - PROD (1-r"
# Re-derived AP vs the artifact's own value, and vs the value the run that
# produced the scores recorded.
_AP_TOL = 1e-12


def _panel() -> tuple[list[str], np.ndarray, dict[str, str]]:
    """The 1689-statement panel, via the compute scripts' own loader.

    Imported rather than re-derived, exactly as the compute scripts do, so this
    test cannot drift from the artifacts it gates. ``load_panel`` resolves its
    inputs relative to the repository root.
    """
    import os
    import sys as _sys

    scripts = str(ROOT / "scripts")
    if scripts not in _sys.path:
        _sys.path.insert(0, scripts)
    from compute_statement_review_queue import load_panel  # noqa: PLC0415

    previous = Path.cwd()
    os.chdir(ROOT)
    try:
        panel = load_panel()
    finally:
        os.chdir(previous)
    # ``load_panel`` returns the whole panel in one object so every downstream
    # block reads one ordering; these tests want the three fields they join on.
    return panel["sids"], panel["y"], panel["matches_hash"]


def _entry_scores(entry: dict, sids: list[str], mhash: dict[str, str]) -> np.ndarray:
    """This entry's probability vector over the panel, from its named file."""
    path = ROOT / entry["scores_path"]
    if entry["kind"] == "paper-literal":
        literal = json.loads(path.read_text())
        table = {
            str(row["stmt_hash"]): float(row["prob_correct"])
            for row in literal["oof_predictions"][entry["scores_key"]]
        }
        return np.array([table[mhash[sid]] for sid in sids])
    table = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        table[row["statement_id"]] = float(row["probability_correct"])
    return np.array([table[sid] for sid in sids])


def test_belief_model_ladder_is_the_artifact_the_manifest_records() -> None:
    """The viewer draws the exact bytes the run manifest signed."""
    manifest = _load(_RUN_MANIFEST_PATH)
    assert manifest["outputs"]["belief_model_ladder"] == _LADDER_PATH.name
    recorded = manifest["output_sha256"][_LADDER_PATH.name]
    digest = hashlib.sha256(_LADDER_PATH.read_bytes()).hexdigest()
    assert digest == recorded


def test_belief_model_ladder_entries_rederive_from_their_named_scores() -> None:
    """Every rung is re-derivable from the prediction file it names."""
    ladder = _load(_LADDER_PATH)
    sids, y, mhash = _panel()
    assert ladder["panel"]["n"] == len(sids)
    assert [(e["label"], e["kind"]) for e in ladder["entries"]] == _LADDER_ENTRIES

    for entry in ladder["entries"]:
        ours = float(average_precision_score(y, _entry_scores(entry, sids, mhash)))
        assert ours == pytest.approx(entry["average_precision"], abs=_AP_TOL), (
            entry["label"]
        )
        # The recorded value travels beside ours, and the agreement flag is the
        # comparison, not a claim.
        assert entry["agrees_with_recorded"] == (
            abs(entry["disagreement_vs_recorded"]) <= _AP_TOL
        ), entry["label"]
        assert entry["disagreement_vs_recorded"] == pytest.approx(
            entry["average_precision"] - entry["recorded_average_precision"],
            abs=_AP_TOL,
        ), entry["label"]


def test_belief_model_ladder_deltas_are_ap_minus_the_marked_baseline() -> None:
    """The bar length IS ap - baseline, and the baseline's own bar is zero."""
    ladder = _load(_LADDER_PATH)
    by_label = {e["label"]: e for e in ladder["entries"]}
    baseline = ladder["baseline"]
    assert baseline["label"] == _LADDER_BASELINE
    assert baseline["average_precision"] == by_label[_LADDER_BASELINE]["average_precision"]

    for entry in ladder["entries"]:
        assert entry["delta_vs_noisy_or_baseline"] == pytest.approx(
            entry["average_precision"] - baseline["average_precision"], abs=_AP_TOL
        ), entry["label"]
    assert by_label[_LADDER_BASELINE]["delta_vs_noisy_or_baseline"] == 0.0

    # The two full-feature rows are the same fitted model reported twice.
    first, second = ladder["checks"]["same_fitted_model_pair"]
    assert by_label[first]["average_precision"] == pytest.approx(
        by_label[second]["average_precision"],
        abs=ladder["checks"]["same_fitted_model_tol"],
    )


def test_belief_model_ladder_guardrails_keep_the_gate_delta_in_context() -> None:
    """+0.048 never travels alone: the against-their-best-model range rides with it."""
    ladder = _load(_LADDER_PATH)
    by_label = {e["label"]: e for e in ladder["entries"]}
    guardrails = ladder["delta_guardrails"]
    gate = guardrails["reading_gate"]

    # Every guardrail number is the ladder's own arithmetic, not a caption.
    assert gate["delta_vs_noisy_or_baseline"] == pytest.approx(
        by_label[gate["label"]]["delta_vs_noisy_or_baseline"], abs=_AP_TOL
    )
    against_best = gate["delta_vs_best_paper_model"]
    for label, delta in against_best.items():
        assert delta == pytest.approx(
            gate["average_precision"] - by_label[label]["average_precision"], abs=_AP_TOL
        ), label
    low, high = gate["delta_vs_best_paper_model_range"]
    assert (low, high) == (min(against_best.values()), max(against_best.values()))
    # The range is strictly tighter than the headline delta measured from the
    # weakest family member — which is the whole point of shipping both.
    assert high < gate["delta_vs_noisy_or_baseline"]

    variant = gate["delta_vs_best_noisy_or_variant"]
    assert variant["delta"] == pytest.approx(
        gate["average_precision"] - by_label[variant["label"]]["average_precision"],
        abs=_AP_TOL,
    )

    proximity = guardrails["reimplementation_proximity"]
    assert proximity["absolute_gap"] == pytest.approx(
        abs(proximity["reimplemented_rf_full_features"]
            - proximity["paper_literal_rf_promoter"]),
        abs=_AP_TOL,
    )
    # Named a consistency check, and the fidelity evidence points elsewhere.
    assert "consistency check" in proximity["status"].lower()
    assert "not fidelity" in proximity["status"].lower()
    assert proximity["fidelity_evidence"]["value"] == pytest.approx(
        _load(_VS_LLMS_PATH)["faithfulness_literal_vs_port"]["pearson_r"], abs=_AP_TOL
    )

    caveats = " ".join(ladder["caveats"])
    assert f"{low:+.4f} to {high:+.4f}" in caveats
    assert "CONSISTENCY CHECK" in caveats
    assert ladder["noisy_or_formula"] == _NOISY_OR_FORMULA
    assert _WRONG_NOISY_OR_FRAGMENT not in _LADDER_PATH.read_text()


def test_non_reading_control_is_the_artifact_the_manifest_records() -> None:
    """The viewer draws the exact bytes the run manifest signed."""
    manifest = _load(_RUN_MANIFEST_PATH)
    assert manifest["outputs"]["non_reading_control"] == _NON_READING_CONTROL_PATH.name
    recorded = manifest["output_sha256"][_NON_READING_CONTROL_PATH.name]
    digest = hashlib.sha256(_NON_READING_CONTROL_PATH.read_bytes()).hexdigest()
    assert digest == recorded


def test_non_reading_control_raw_row_is_the_shipped_noisy_or() -> None:
    """The raw row IS SimpleScorer, and the control lands below it."""
    control = _load(_NON_READING_CONTROL_PATH)
    rows = {row["key"]: row for row in control["rows"]}
    raw = rows[control["baseline_row"]]
    full = rows[control["control_row"]]

    # The raw row reproduces the shipped SimpleScorer predictions bit-exactly and
    # its AP is the value that run's manifest recorded.
    checks = control["checks"]
    assert checks["raw_row_reproduces_simple_scorer_bit_exactly"] == control["panel"]["n"]
    assert checks["raw_row_max_abs_delta_vs_simple_scorer"] == 0.0
    recorded = checks["raw_row_average_precision_vs_recorded"]
    assert recorded["ours"] == raw["average_precision"]
    assert recorded["ours"] == pytest.approx(recorded["recorded"], abs=_AP_TOL)
    manifest = json.loads((ROOT / recorded["recorded_in"]).read_text())
    assert recorded["recorded"] == (
        manifest["diagnostic_metrics"][recorded["recorded_key"]]["pooled_average_precision"]
    )

    # THE finding: the three non-reading subtractions, applied with no model
    # verdict at all, land BELOW the ungated noisy-OR.
    assert full["average_precision"] < raw["average_precision"]
    assert control["control_lands_below_raw"] is True
    assert control["control_minus_raw_average_precision"] == pytest.approx(
        full["average_precision"] - raw["average_precision"], abs=_AP_TOL
    )
    assert control["control_minus_raw_average_precision"] < 0.0
    for row in control["rows"]:
        assert row["delta_vs_raw_noisy_or"] == pytest.approx(
            row["average_precision"] - raw["average_precision"], abs=_AP_TOL
        ), row["key"]

    # Reading is the contrast, and it sits well above both.
    contrast = control["contrast"]
    assert contrast["average_precision"] > raw["average_precision"]
    assert contrast["delta_vs_full_control"] == pytest.approx(
        contrast["average_precision"] - full["average_precision"], abs=_AP_TOL
    )

    # A reader can only ever remove belief — checked over all 1689 x 4.
    subtractive = control["subtractive_check"]
    assert subtractive["n_exceeding_noisy_or"] == 0
    assert subtractive["n_comparisons"] == control["panel"]["n"] * len(subtractive["arms"])
    assert sum(a["n_exceeding_noisy_or"] for a in subtractive["arms"].values()) == 0

    assert control["noisy_or_formula"] == _NOISY_OR_FORMULA
    assert _WRONG_NOISY_OR_FRAGMENT not in _NON_READING_CONTROL_PATH.read_text()


def test_non_reading_control_census_and_dedup_are_internally_consistent() -> None:
    """The census and the de-dup arithmetic close against the rows they explain."""
    control = _load(_NON_READING_CONTROL_PATH)
    census = control["route_census"]
    dedup = control["dedup"]
    rows = {row["key"]: row for row in control["rows"]}

    # Every route is accounted for, twice: once per unique pair, once weighted.
    assert sum(census["routes"].values()) == census["n_unique_pairs"]
    assert set(census["routes"]) == set(census["routes_multiplicity_weighted"])
    assert census["no_text"] == census["routes"]["no_text"]
    assert census["deterministic_mismatch"] == census["routes"]["deterministic_mismatch"]
    assert census["deterministic_pseudogene"] == census["routes"]["deterministic_pseudogene"]
    assert set(census["readable_routes"]) == set(census["routes"]) - {
        "no_text", "deterministic_mismatch", "deterministic_pseudogene"
    }

    # De-dup: unique + excess == the multiplicity the paper's noisy-OR counts.
    assert dedup["n_unique_pairs"] == census["n_unique_pairs"]
    assert (
        dedup["n_unique_pairs"] + dedup["excess_pairs"] == dedup["n_summed_multiplicity"]
    )
    assert 0 < dedup["statements_with_excess"] <= control["panel"]["n"]
    assert dedup["excess_pairs"] >= dedup["statements_with_excess"]

    # Each row's evidence count is the census minus exactly the routes it drops.
    assert rows["raw"]["n_evidence_scored"] == dedup["n_summed_multiplicity"]
    for row in control["rows"]:
        if row["weight"] != "unique_pair":
            continue
        dropped = sum(census["routes"][route] for route in row["dropped_routes"])
        assert row["n_evidence_scored"] == census["n_unique_pairs"] - dropped, row["key"]

    # The memo-reported production de-dup scope is flagged as not re-derived.
    scope = control["production_dedup_scope_difference"]
    assert "NOT re-derived" in scope["status"]
    assert scope["excess_pairs"] > dedup["excess_pairs"]
    assert scope["statements_with_excess"] > dedup["statements_with_excess"]


# --- framing correction (scripts/compute_framing_correction.py) -------------------
# The panel that runs BEFORE the head-to-head, establishing that the reader arm is
# the paper's own noisy-OR on a filtered evidence set. Three legs are gated here:
#   (b) RE-DERIVED independently from the four 1689-row prediction files and the
#       shipped SimpleScorer scores, then asserted equal to the artifact — this is
#       the headline, so pytest reproduces it rather than reading it;
#   (a) asserted against what the four bundle manifests ACTUALLY say, plus the
#       sha256 of the aggregation config and the two implementation sources;
#   (c) the artifact's own internal arithmetic, closed.
# The reachable-set enumeration is NOT re-run here: following this file's stated
# precedent, expensive re-derivation is the compute script's own assertion to own
# (it exits non-zero on a counterexample), not pytest's.

_FRAMING_PATH = _MODEL_DIR / "framing_correction.json"
_GOLD_PATH = (ROOT / "data" / "results" / "indra_paper_statement_gold_20260717"
              / "paper_statement_gold.jsonl")
_AGGREGATION_PATH = ROOT / "data" / "comparison" / "aggregation.json"
_MODELS_DIR = ROOT / "data" / "comparison" / "models"
_NOISE_MODEL_PATH = ROOT / "src" / "indra_belief" / "noise_model.py"
_STATEMENT_BELIEF_PATH = ROOT / "src" / "indra_belief" / "statement_belief.py"
_SIMPLE_SCORER_PREDICTIONS = (
    ROOT / "data" / "results" / "current_indra_simple_paper_20260717"
    / "current_indra_simple_default_predictions.jsonl"
)
_REQUIRED_AGGREGATION = "indra_default_hard_gate"
# Fixed presentation order, matching the compute script's READER_ARMS.
_FRAMING_ARMS = ["gemma_4_26b", "glm_5", "gemma_4_31b", "gemma_4_e2b"]


def _jsonl_scores(path: Path) -> dict[str, float]:
    table = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        table[row["statement_id"]] = float(row["probability_correct"])
    return table


def test_framing_correction_is_the_artifact_the_manifest_records() -> None:
    """The viewer draws the exact bytes the run manifest signed."""
    manifest = _load(_RUN_MANIFEST_PATH)
    assert manifest["outputs"]["framing_correction"] == _FRAMING_PATH.name
    recorded = manifest["output_sha256"][_FRAMING_PATH.name]
    digest = hashlib.sha256(_FRAMING_PATH.read_bytes()).hexdigest()
    assert digest == recorded
    # The P1 outputs this artifact was added beside are untouched.
    for name in ("ap_decomposition_by_paper_band.json", "statement_review_queue.json",
                 "belief_model_ladder.json", "non_reading_control.json"):
        assert hashlib.sha256((_MODEL_DIR / name).read_bytes()).hexdigest() == (
            manifest["output_sha256"][name]
        ), name


def test_framing_correction_subtractive_leg_rederives_from_the_prediction_files() -> None:
    """THE headline, re-derived: no reader belief exceeds the noisy-OR.

    Four 1689-row prediction files against one 1689-row SimpleScorer file — cheap
    enough to reproduce outright rather than to read off the artifact.
    """
    framing = _load(_FRAMING_PATH)
    sids, _y, _mhash = _panel()
    n = len(sids)

    simple = _jsonl_scores(_SIMPLE_SCORER_PREDICTIONS)
    assert set(simple) == set(sids)
    noisy_or = np.array([simple[sid] for sid in sids])

    total_exceeding = 0
    for arm in _FRAMING_ARMS:
        scores = _jsonl_scores(_MODELS_DIR / arm / "all_source_predictions.jsonl")
        assert set(scores) == set(sids), arm
        p = np.array([scores[sid] for sid in sids])

        exceeding = int((p > noisy_or).sum())
        at_zero = int((p == 0.0).sum())
        nonzero = int((p != 0.0).sum())
        max_above = float(np.max(p - noisy_or))
        total_exceeding += exceeding

        recorded = framing["subtractive"]["arms"][arm]
        assert recorded["n_statements"] == n, arm
        assert recorded["n_exceeding_noisy_or"] == exceeding == 0, arm
        assert recorded["n_at_exactly_zero"] == at_zero, arm
        assert recorded["n_nonzero"] == nonzero, arm
        assert recorded["max_belief_above_noisy_or"] == pytest.approx(max_above, abs=_AP_TOL), arm
        assert at_zero + nonzero == n, arm

        # The reachable-value leg counts the same zero block on the same arm.
        assert framing["reachable_values"]["arms"][arm]["n_at_exactly_zero"] == at_zero, arm

    assert framing["subtractive"]["n_exceeding_noisy_or"] == total_exceeding == 0
    assert framing["subtractive"]["n_comparisons"] == n * len(_FRAMING_ARMS)

    # The same finding lives in P1's artifact; the framing artifact pins its bytes.
    control = _load(_NON_READING_CONTROL_PATH)
    assert framing["subtractive"]["cross_check"]["sha256"] == (
        hashlib.sha256(_NON_READING_CONTROL_PATH.read_bytes()).hexdigest()
    )
    for arm in _FRAMING_ARMS:
        theirs = control["subtractive_check"]["arms"][arm]
        ours = framing["subtractive"]["arms"][arm]
        for field in ("n_statements", "n_exceeding_noisy_or", "n_at_exactly_zero",
                      "max_belief_above_noisy_or"):
            assert theirs[field] == ours[field], f"{arm}.{field}"


def test_framing_correction_declaration_is_what_the_manifests_actually_say() -> None:
    """Leg (a) is checked against the bundles and the source bytes, not transcribed."""
    framing = _load(_FRAMING_PATH)
    declaration = framing["declaration"]
    assert declaration["required_aggregation"] == _REQUIRED_AGGREGATION
    assert [arm["arm"] for arm in declaration["arms"]] == _FRAMING_ARMS

    aggregation_sha = hashlib.sha256(_AGGREGATION_PATH.read_bytes()).hexdigest()
    assert declaration["aggregation_config"]["sha256"] == aggregation_sha
    sources = {
        "noise_model": (_NOISE_MODEL_PATH,
                        hashlib.sha256(_NOISE_MODEL_PATH.read_bytes()).hexdigest()),
        "statement_belief": (_STATEMENT_BELIEF_PATH,
                             hashlib.sha256(_STATEMENT_BELIEF_PATH.read_bytes()).hexdigest()),
    }
    for key, (path, digest) in sources.items():
        assert declaration["implementation_sources"][key]["sha256"] == digest, key
        assert declaration["implementation_sources"][key]["path"] == str(
            path.relative_to(ROOT)
        ), key

    for arm in declaration["arms"]:
        bundle = _load(ROOT / arm["manifest_path"])
        notes = bundle["implementation"]["notes"]
        # Everything the panel prints is what the bundle itself says.
        assert notes["aggregation"] == arm["aggregation"] == _REQUIRED_AGGREGATION, arm["arm"]
        assert notes["reader_profile"] is None and arm["reader_profile"] is None, arm["arm"]
        assert notes["dedup"] == arm["dedup"], arm["arm"]
        assert bundle["implementation"]["implementation"] == arm["implementation"], arm["arm"]
        # ... and what it says matches the bytes in the tree.
        assert notes["inputs"]["aggregation_config"]["sha256"] == aggregation_sha, arm["arm"]
        assert arm["declared_aggregation_config_sha256"] == aggregation_sha, arm["arm"]
        assert arm["aggregation_config_sha256_matches"] is True, arm["arm"]
        assert arm["implementation_component_sha256_matches"] is True, arm["arm"]
        for key, (_path, digest) in sources.items():
            assert notes["implementation_components"][key] == digest, f"{arm['arm']}.{key}"
            assert arm["implementation_component_sha256"][key] == digest, f"{arm['arm']}.{key}"

    # The priors are the panel's own census, and the groups partition it.
    aggregation = _load(_AGGREGATION_PATH)
    assert aggregation["aggregation"] == _REQUIRED_AGGREGATION
    assert aggregation["reader_profile"] is None
    grouped = []
    for group in declaration["prior_groups"]:
        grouped.extend(group["sources"])
        for source in group["sources"]:
            rand, syst = aggregation["priors"][source]
            assert (group["rand"], group["syst"]) == (rand, syst), source
    assert sorted(grouped) == sorted(declaration["panel_priors"])
    for source, prior in declaration["panel_priors"].items():
        rand, syst = aggregation["priors"][source]
        assert (prior["rand"], prior["syst"]) == (rand, syst), source


def test_framing_correction_negative_breakdown_matches_the_gold() -> None:
    """452 = 341 adjudication-safe + 111 flagged, re-derived from the gold jsonl."""
    framing = _load(_FRAMING_PATH)
    safe = flagged = 0
    for line in _GOLD_PATH.read_text().splitlines():
        if not line.strip():
            continue
        policy = json.loads(line).get("paper_replication_policy") or {}
        if policy.get("released_paper_correct") is None:
            continue
        if int(policy["released_paper_correct"]) != 0:
            continue
        if policy.get("label_is_adjudication_safe") is True:
            safe += 1
        else:
            flagged += 1

    panel = framing["panel"]
    breakdown = panel["negative_breakdown"]
    assert breakdown["adjudication_safe_negatives"] == safe
    assert breakdown["flagged_label_is_adjudication_safe_false"] == flagged
    assert breakdown["n_errors"] == panel["n_errors"] == safe + flagged
    assert panel["n_errors"] + panel["n_correct"] == panel["n"]
    assert panel["label"] == "paper_replication_policy.released_paper_correct"
    # The same breakdown P1's ladder carries; the two panels are one panel.
    assert breakdown == _load(_LADDER_PATH)["panel"]["negative_breakdown"]


def test_framing_correction_internal_arithmetic_closes() -> None:
    """Leg (c)'s tiers partition the non-zero scores, and the floor is below them."""
    framing = _load(_FRAMING_PATH)
    panel_n = framing["panel"]["n"]
    reachable = framing["reachable_values"]
    search = reachable["search"]
    null = reachable["null_baseline"]

    assert search["n_counterexamples"] == 0
    assert search["max_nodes_used"] <= search["node_budget_per_statement"]

    for arm in _FRAMING_ARMS:
        row = reachable["arms"][arm]
        assert row["n_at_exactly_zero"] + row["n_nonzero"] == panel_n, arm
        assert (
            row["n_confirmed_reachable"] + row["n_budget_exhausted"] + row["n_counterexamples"]
            == row["n_nonzero"]
        ), arm
        assert row["n_counterexamples"] == 0, arm
        # The tighter tier is a subset of the looser one.
        assert row["n_bit_exact"] <= row["n_confirmed_reachable"], arm
        assert row["share_confirmed"] == pytest.approx(
            row["n_confirmed_reachable"] / row["n_nonzero"], abs=_PARITY_TOL), arm
        assert row["share_bit_exact"] == pytest.approx(
            row["n_bit_exact"] / row["n_nonzero"], abs=_PARITY_TOL), arm
        # Anything the search left unsettled is NAMED, never folded into confirmed.
        assert len(row["unresolved_statement_ids"]) == (
            row["n_budget_exhausted"] + row["n_counterexamples"]), arm

        # The claim is quoted against a floor, and the floor is well below it.
        floor = null["arms"][arm]
        assert floor["permuted_rate_min"] <= floor["permuted_rate_mean"] <= floor["permuted_rate_max"], arm
        assert floor["permuted_rate_mean"] < row["share_confirmed"], arm
        assert 0 < floor["n_enumerable_nonzero"] <= row["n_nonzero"], arm
        assert len(floor["permuted_rates"]) == null["n_permutations"], arm
        assert floor["permuted_rate_mean"] == pytest.approx(
            sum(floor["permuted_rates"]) / len(floor["permuted_rates"]), abs=_PARITY_TOL), arm
        assert floor["permuted_rate_min"] == min(floor["permuted_rates"]), arm
        assert floor["permuted_rate_max"] == max(floor["permuted_rates"]), arm

    assert 0 < null["n_statements_enumerable"] <= null["n_statements_on_panel"] == panel_n
    pooled = [rate for arm in _FRAMING_ARMS for rate in null["arms"][arm]["permuted_rates"]]
    assert null["pooled_permuted_rate_min"] == min(pooled)
    assert null["pooled_permuted_rate_max"] == max(pooled)
    assert null["pooled_permuted_rate_mean"] == pytest.approx(
        sum(pooled) / len(pooled), abs=_PARITY_TOL)

    # SimpleScorer's own floor on this panel, so a reader zero is a different object.
    floor = reachable["noisy_or_floor_on_panel"]["value"]
    scores = _jsonl_scores(_SIMPLE_SCORER_PREDICTIONS)
    assert floor == min(scores.values()) > 0.0

    # The formula, stated correctly and nowhere stated wrongly.
    assert framing["noisy_or_formula"] == _NOISY_OR_FORMULA
    assert framing["aggregation"] == _REQUIRED_AGGREGATION
    assert _WRONG_NOISY_OR_FRAGMENT not in _FRAMING_PATH.read_text()

    # The script's own assertions travel as checks, and none of them may be False.
    for key, value in framing["checks"].items():
        if key == "note":
            continue
        assert value is True, key
