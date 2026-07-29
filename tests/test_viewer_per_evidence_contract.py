"""Gate the /paper PER-EVIDENCE surface from the Python side.

The .mjs runner asserts the TypeScript data contract against the shipped
artifact. This is the other end of that parity: it RE-DERIVES the artifact's
load-bearing numbers from the source data rather than reading them back off the
artifact, so the figure cannot drift in either language.

WHAT IS RE-DERIVED HERE
  * The panel itself. The 5,379 reviewed evidence pairs are rebuilt from
    ``paper_evidence_adjudication.jsonl`` AND, independently, from
    ``paper_statement_gold.jsonl``'s own ``evidence_review`` blocks. Two files,
    one truth; the artifact's census must equal both.
  * Every aggregate metric, recomputed with scikit-learn from the shipped
    per-pair table (``per_evidence_pairs.jsonl``, 4 MB). This is the check that
    the headline AUROC is actually what the 5,379 shipped per-pair scores say.
  * INDRA's bundled single-evidence belief, recomputed as ``1 - (rand + syst)``
    from ``aggregation.json`` — the quantity the shared-prior register draws.
  * The chi-square behind the shared-prior defect, from the per-source counts.
  * Each arm's statement-grain AUROC, from its own 1,689-row prediction file.
  * The evidence-grain contamination check, redone from the prompt manifest and
    the panel corpus.

WHAT IS NOT RE-DERIVED, AND WHY. The recovery of per-evidence verdicts from the
7.4 GB of raw provider attempts, and the statement-grain reconciliation that
proves those verdicts rebuild the shipped statement probabilities, are the
compute script's own assertions to own — it exits non-zero on either. Same
precedent as the 19 MB execution map in ``test_viewer_paper_literal_contract``.
What pytest holds is that the SHIPPED per-pair table reproduces the SHIPPED
aggregates, which is the seam a rerun could actually break.

MISATTRIBUTION is gated here too, because it is the claim most costly to get
wrong in front of this audience: no arm on this plate may be presented as a
published 2023 method, and every arm must carry a provenance sentence.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
TS_RUNNER = ROOT / "viewer" / "scripts" / "test-per-evidence-contract.mjs"

ARTIFACT_DIR = ROOT / "data" / "results" / "per_evidence_comparison_20260727"
ARTIFACT_PATH = ARTIFACT_DIR / "per_evidence_comparison.json"
PAIRS_PATH = ARTIFACT_DIR / "per_evidence_pairs.jsonl"

EVIDENCE_GOLD = (
    ROOT / "data/results/indra_paper_statement_gold_20260717/paper_evidence_adjudication.jsonl"
)
STATEMENT_GOLD = (
    ROOT / "data/results/indra_paper_statement_gold_20260717/paper_statement_gold.jsonl"
)
EXECUTION_MAP = (
    ROOT / "data/benchmark/indra_paper_unique_pairs_20260717_execution_map.jsonl"
)
AGGREGATION = ROOT / "data/comparison/aggregation.json"
PROMPT_MANIFEST = ROOT / "data/comparison/grounding_replay/manifest.json"
PANEL_CORPUS = ROOT / "data/corpora/indra_paper_unique_pairs_20260717_statements.json"

# The figure's frozen expectations. Changing any of these is a decision, not a
# refactor, so they are pinned from both languages.
EXPECTED_REVIEWED_PAIRS = 5379
EXPECTED_STATEMENTS = 1689
# Recomputed-vs-shipped tolerance. Both sides are float64 sklearn calls on the
# same vectors, so this is float noise, not a fudge factor.
METRIC_TOL = 1e-9

pytestmark = pytest.mark.skipif(
    not ARTIFACT_PATH.is_file(),
    reason="per-evidence comparison artifact not present; run scripts/compute_per_evidence_comparison.py",
)


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT_PATH.read_text())


@pytest.fixture(scope="module")
def pairs() -> list[dict]:
    return _load_jsonl(PAIRS_PATH)


@pytest.fixture(scope="module")
def gold() -> dict[tuple[str, str], int]:
    """The reviewed panel, from the per-evidence adjudication file."""
    out: dict[tuple[str, str], int] = {}
    for record in _load_jsonl(EVIDENCE_GOLD):
        if record["review_status"] == "unreviewed":
            continue
        key = (str(record["paper_statement_hash"]), str(record["source_hash"]))
        assert key not in out, f"{key}: duplicate reviewed pair"
        out[key] = record["evidence_gold_label"]
    return out


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_viewer_per_evidence_contract() -> None:
    completed = subprocess.run(
        ["node", "--experimental-strip-types", str(TS_RUNNER)],
        cwd=ROOT / "viewer",
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, (
        "viewer per-evidence contract assertions failed:\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def test_panel_is_the_same_5379_pairs_in_two_independent_files(
    artifact: dict, gold: dict
) -> None:
    """The reviewed panel must agree between the adjudication file and the
    statement gold's own ``evidence_review`` blocks. A drift in either would move
    the denominator under every number on the plate."""
    cross: dict[tuple[str, str], int] = {}
    statements = _load_jsonl(STATEMENT_GOLD)
    for record in statements:
        stmt = str(record["paper_statement_hash"])
        review = record["evidence_review"]
        for source_hash in review["positive_source_hashes"]:
            cross[(stmt, str(source_hash))] = 1
        for source_hash in review["negative_source_hashes"]:
            cross[(stmt, str(source_hash))] = 0

    assert cross == gold
    assert len(gold) == EXPECTED_REVIEWED_PAIRS
    assert len(statements) == EXPECTED_STATEMENTS

    panel = artifact["panel"]
    assert panel["n_reviewed_pairs"] == len(gold)
    assert panel["n_positive"] == sum(gold.values())
    assert panel["n_negative"] == len(gold) - sum(gold.values())
    assert panel["n_statements"] == EXPECTED_STATEMENTS
    # Every statement contributes at least one reviewed pair; a panel that covered
    # only some statements would not be the statement panel's evidence.
    assert panel["n_statements_with_reviewed_pair"] == EXPECTED_STATEMENTS
    assert len({key[0] for key in gold}) == EXPECTED_STATEMENTS


def test_shipped_pair_table_is_the_panel(pairs: list[dict], gold: dict) -> None:
    """The audit trail must BE the panel: same keys, same labels, no extras."""
    assert len(pairs) == EXPECTED_REVIEWED_PAIRS
    table = {
        (row["paper_statement_hash"], row["source_hash"]): row["gold_label"] for row in pairs
    }
    assert len(table) == len(pairs), "duplicate keys in the shipped pair table"
    assert table == gold

    execution = {
        (str(r["paper_statement_hash"]), str(r["source_hash"])): r
        for r in _load_jsonl(EXECUTION_MAP)
    }
    for row in pairs:
        key = (row["paper_statement_hash"], row["source_hash"])
        assert key in execution, f"{key}: not in the execution map"
        assert row["source_api"] == execution[key]["source_api"]


def test_shipped_aggregates_are_what_the_pair_table_says(
    artifact: dict, pairs: list[dict]
) -> None:
    """Recompute every drawn per-evidence metric from the shipped per-pair scores.

    This is the seam a rerun breaks: an arm's aggregate could survive while its
    per-pair vector changed, or vice versa. Both must agree, on the same 5,379
    rows, under scikit-learn's own tie-aware estimators.
    """
    labels = np.array([row["gold_label"] for row in pairs], dtype=np.int64)
    assert set(np.unique(labels)) == {0, 1}

    drawn = [arm for arm in artifact["arms"] if "metrics" in arm]
    assert drawn, "no arm carries metrics"
    for arm in drawn:
        arm_id = arm["id"]
        assert arm_id in pairs[0], f"{arm_id}: no per-pair score column"
        scores = np.array([row[arm_id] for row in pairs], dtype=np.float64)
        assert np.isfinite(scores).all()

        assert roc_auc_score(labels, scores) == pytest.approx(
            arm["metrics"]["auroc"], abs=METRIC_TOL
        ), f"{arm_id}: shipped AUROC is not what its per-pair scores say"
        assert average_precision_score(labels, scores) == pytest.approx(
            arm["metrics"]["average_precision_correct"], abs=METRIC_TOL
        ), f"{arm_id}: shipped AP is not what its per-pair scores say"
        # Error detection: positive class is the INCORRECT evidence pair.
        assert average_precision_score(1 - labels, -scores) == pytest.approx(
            arm["metrics"]["average_precision_incorrect"], abs=METRIC_TOL
        ), f"{arm_id}: shipped error-class AP is not what its per-pair scores say"
        assert int(np.unique(scores).size) == arm["metrics"]["distinct_scores"]

        # Error-detection F1 at the reader's own decision boundary.
        threshold = artifact["decision_threshold"]
        predicted_incorrect = scores < threshold
        truly_incorrect = labels == 0
        tp = int((predicted_incorrect & truly_incorrect).sum())
        fp = int((predicted_incorrect & ~truly_incorrect).sum())
        fn = int((~predicted_incorrect & truly_incorrect).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        assert f1 == pytest.approx(
            arm["metrics"]["error_detection"]["f1"], abs=METRIC_TOL
        ), f"{arm_id}: shipped error-detection F1 disagrees with its own per-pair verdicts"

        # The bootstrap interval must contain its own point estimate; the figure
        # draws the interval as a whisker THROUGH the mark.
        interval = arm["metrics"]["interval"]["auroc"]
        assert interval["ci95_low"] <= arm["metrics"]["auroc"] <= interval["ci95_high"]

    assert artifact["n_bootstrap"] == 10000
    assert artifact["seed"] == 20260717


def test_per_source_strata_are_the_census(artifact: dict, pairs: list[dict]) -> None:
    """Every per-source AUROC on the plate is recomputed from the same rows."""
    labels = {row["source_api"]: [] for row in pairs}
    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in pairs:
        labels[row["source_api"]].append(row["gold_label"])
        for arm in artifact["arms"]:
            if "metrics" not in arm:
                continue
            scores[arm["id"]][row["source_api"]].append(row[arm["id"]])

    census = {row["source"]: row for row in artifact["coverage"]["sources"]}
    assert sum(row["reviewed_pairs"] for row in census.values()) == EXPECTED_REVIEWED_PAIRS
    for source, values in labels.items():
        assert census[source]["reviewed_pairs"] == len(values)
        assert census[source]["positive_pairs"] == sum(values)
        assert census[source]["negative_pairs"] == len(values) - sum(values)
        assert census[source]["observed_correct_fraction"] == pytest.approx(
            sum(values) / len(values), abs=1e-12
        )

    for arm in artifact["arms"]:
        if "metrics" not in arm:
            continue
        for source, block in arm["per_source"].items():
            y = np.array(labels[source], dtype=np.int64)
            p = np.array(scores[arm["id"]][source], dtype=np.float64)
            assert block["n"] == y.size
            assert roc_auc_score(y, p) == pytest.approx(
                block["auroc"], abs=METRIC_TOL
            ), f"{arm['id']}/{source}: shipped stratum AUROC is not what its rows say"


def test_bundled_single_evidence_prior_and_the_shared_prior_defect(artifact: dict) -> None:
    """INDRA's own single-evidence belief, and the spread that one number covers.

    ``SimpleScorer`` on one evidence from one source reduces to
    ``1 - (syst_s + rand_s)``, so the register's tick positions are recomputed
    straight from the aggregation config the run used. The chi-square is
    recomputed from the per-source counts, so the defect claim rests on this
    panel's own numbers rather than on a field.
    """
    priors = json.loads(AGGREGATION.read_text())["priors"]
    census = {row["source"]: row for row in artifact["coverage"]["sources"]}
    for source, row in census.items():
        rand, syst = priors[source]
        assert row["bundled_prior_at_one_evidence"] == pytest.approx(
            1.0 - (rand + syst), abs=1e-12
        ), f"{source}: the drawn prior is not INDRA's own single-evidence belief"

    blocks = artifact["shared_prior_defect"]["blocks"]
    assert blocks, "the shared-prior block is what the per-source register draws"
    block = blocks[0]
    names = [entry["source"] for entry in block["sources"]]
    assert len(names) >= 2
    # One prior, many behaviours: that is the whole claim.
    assert len({census[name]["bundled_prior_at_one_evidence"] for name in names}) == 1
    observed = [census[name]["observed_correct_fraction"] for name in names]
    assert block["observed_correct_fraction_min"] == pytest.approx(min(observed), abs=1e-12)
    assert block["observed_correct_fraction_max"] == pytest.approx(max(observed), abs=1e-12)

    table = np.array(
        [[census[name]["positive_pairs"], census[name]["negative_pairs"]] for name in names],
        dtype=np.float64,
    )
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    chi2 = float(((table - expected) ** 2 / expected).sum())
    assert chi2 == pytest.approx(block["chi2"], rel=1e-9)
    assert block["dof"] == (table.shape[0] - 1) * (table.shape[1] - 1)
    assert block["p_value"] < 1e-6

    # The defect is measured elsewhere on this site against a DIFFERENT gold, and
    # reports a different spread. The artifact must say so rather than let two
    # numbers for one defect look like a contradiction.
    assert "panel" in artifact["shared_prior_defect"]["note"]


def test_statement_grain_marks_come_from_the_same_models(artifact: dict) -> None:
    """Each lane's statement mark is recomputed from that arm's own predictions."""
    statements = _load_jsonl(STATEMENT_GOLD)
    ids = [r["canonical_corpus"]["statement_id"] for r in statements]
    labels = np.array(
        [r["paper_replication_policy"]["released_paper_correct"] for r in statements],
        dtype=np.int64,
    )
    assert artifact["statement_grain"]["positive_rate"] == pytest.approx(
        float(labels.mean()), abs=1e-12
    )

    for arm in artifact["arms"]:
        if "statement_grain" not in arm:
            continue
        block = arm["statement_grain"]
        path = ROOT / block["predictions_path"]
        assert path.is_file(), f"{arm['id']}: {block['predictions_path']} is missing"
        table = {
            r["statement_id"]: float(r["probability_correct"]) for r in _load_jsonl(path)
        }
        scores = np.array([table[s] for s in ids], dtype=np.float64)
        assert block["n_statements"] == len(ids)
        assert roc_auc_score(labels, scores) == pytest.approx(
            block["auroc"], abs=METRIC_TOL
        ), f"{arm['id']}: shipped statement AUROC is not what its prediction file says"
        assert average_precision_score(labels, scores) == pytest.approx(
            block["average_precision"], abs=METRIC_TOL
        )

    # The two marks in a lane are only the same model because the per-evidence
    # verdicts rebuild the shipped statement probability EXACTLY. The compute
    # script owns that reconciliation; here we hold that it was run and passed.
    reagg = artifact["reaggregation"]
    assert reagg["verified"] is True
    assert reagg["arms"], "no arm was reconciled"
    for arm_id, block in reagg["arms"].items():
        assert block["n_exact"] == block["n_statements"] == EXPECTED_STATEMENTS, arm_id
        assert block["max_abs_diff"] == 0.0, arm_id

    note = artifact["statement_grain"]["note"]
    assert "not a causal increment" in note
    assert "not paired" in note


def test_misattribution_ban(artifact: dict) -> None:
    """No arm on this plate may be presented as a published 2023 paper method."""
    for arm in artifact["arms"]:
        attribution = arm.get("attribution")
        assert isinstance(attribution, str) and attribution.strip(), (
            f"{arm['id']}: every arm must carry a provenance sentence"
        )

    by_id = {arm["id"]: arm for arm in artifact["arms"]}
    assert "is not that arm" in by_id["indra-default-source-prior"]["attribution"]
    assert "UNFITTED" in by_id["indra-default-source-prior"]["attribution"]
    for arm_id in ("indra-bayes-source-oof", "indra-bayes-subtype-oof"):
        assert "publishes NO" in by_id[arm_id]["attribution"], arm_id
    for arm in artifact["arms"]:
        if arm["kind"] == "reader":
            assert "NOT zero-shot" in arm["attribution"], arm["id"]

    # No arm may be NAMED after a paper method. The paper's own Table 6 rows are
    # statement-grain only, and the exclusion must ship as data.
    for arm in artifact["arms"]:
        assert not re.search(r"\bRF\b|Belief Orig|Log LR|KNN|SVC", arm["display"]), arm["display"]
    excluded = artifact["coverage"]["excluded_baselines"]
    assert excluded, "the statement-only exclusion must ship as data, not as prose"
    assert all(entry["family"] and entry["reason"] for entry in excluded)


def test_coverage_census_hides_nothing(artifact: dict) -> None:
    coverage = artifact["coverage"]
    assert (
        coverage["reviewed_pairs"] + coverage["unreviewed_pairs"]
        == coverage["executed_unique_pairs"]
    )
    assert coverage["reviewed_pairs"] == EXPECTED_REVIEWED_PAIRS
    for arm_id, block in coverage["per_arm"].items():
        # A reviewed pair with no reader verdict would put one arm on a different
        # panel from the others; the compute script gates on it and so does this.
        assert block["reviewed_pairs_unscored"] == 0, arm_id
        assert block["reviewed_pairs_scored"] == EXPECTED_REVIEWED_PAIRS, arm_id
        assert block["raw_attempts_sha256_matches_manifest"] is True, arm_id
        assert sum(block["tier_census_reviewed"].values()) == EXPECTED_REVIEWED_PAIRS, arm_id
        # `no_text` rows were never semantically read, yet they carry a verdict.
        # They are a small, visible part of the panel, not a hidden one.
        assert "no_text" in block["tier_census_reviewed"], arm_id


def test_contamination_at_evidence_grain_is_redone_here(artifact: dict) -> None:
    """The (agent set, type) check is not enough at this grain — redo it.

    ``tests/test_paper_panel_fewshot_disjoint.py`` establishes disjointness at
    (agent set, statement type) grain. The unit here is the evidence PAIR, so the
    demonstration sentences are matched against the reviewed pairs' own corpus
    text, independently of the artifact.
    """
    manifest = json.loads(PROMPT_MANIFEST.read_text())
    sentences: set[str] = set()
    for prefix in manifest["prompt_components"]["main_message_prefixes"]:
        for message in prefix["messages"]:
            if message.get("role") != "user":
                continue
            found = re.search(r"EVIDENCE:\s*(.+)", message["content"])
            if found:
                sentences.add(" ".join(found.group(1).casefold().split()))
    assert sentences, "parsed no demonstration sentences — the check would be vacuous"

    corpus = json.loads(PANEL_CORPUS.read_text())
    statements = _load_jsonl(STATEMENT_GOLD)
    assert len(corpus) == len(statements)
    reviewed: set[tuple[str, str]] = set()
    for record in statements:
        stmt = str(record["paper_statement_hash"])
        review = record["evidence_review"]
        for source_hash in review["positive_source_hashes"] + review["negative_source_hashes"]:
            reviewed.add((stmt, str(source_hash)))

    hits: set[tuple[str, str]] = set()
    scanned = 0
    panel_tokens: set[str] = set()
    for record, statement in zip(statements, corpus):
        stmt = str(record["paper_statement_hash"])
        for evidence in statement["evidence"]:
            key = (stmt, str(evidence["source_hash"]))
            if key not in reviewed or not evidence.get("text"):
                continue
            scanned += 1
            normalised = " ".join(evidence["text"].casefold().split())
            panel_tokens.update(normalised.split())
            if normalised in sentences:
                hits.add(key)

    # Non-vacuity: a null intersection between two vocabularies that never meet
    # would prove nothing about leakage.
    demo_tokens = {token for sentence in sentences for token in sentence.split()}
    assert demo_tokens & panel_tokens, "the two vocabularies do not intersect at all"

    shipped = artifact["contamination"]
    assert shipped["n_demonstration_sentences"] == len(sentences)
    assert shipped["n_reviewed_pairs_scanned"] == scanned
    assert shipped["n_overlapping_pairs"] == len(hits)
    assert {tuple(pair) for pair in shipped["overlapping_pairs"]} == hits

    # Whatever the count, it must ship with a sensitivity beside it and the
    # primary panel must keep every reviewed pair.
    if hits:
        sensitivity = shipped["sensitivity"]
        assert sensitivity["n_pairs_excluded"] == len(hits)
        assert sensitivity["n_pairs_kept"] == EXPECTED_REVIEWED_PAIRS - len(hits)
        assert artifact["panel"]["n_reviewed_pairs"] == EXPECTED_REVIEWED_PAIRS
        assert shipped["n_overlapping_pairs_same_claim"] <= shipped["n_overlapping_pairs"]


def test_power_claim_ships_with_its_caveat(artifact: dict) -> None:
    """The 3.2x item ratio is real; the effective power gain is smaller.

    Evidence pairs within a statement are not independent, and the bootstrap
    resamples pairs rather than statements. Shipping the ratio without that
    sentence would overclaim, so the sentence is gated.
    """
    power = artifact["power"]
    assert power["n_evidence_pairs"] == EXPECTED_REVIEWED_PAIRS
    assert power["n_statements"] == EXPECTED_STATEMENTS
    assert power["ratio"] == pytest.approx(
        EXPECTED_REVIEWED_PAIRS / EXPECTED_STATEMENTS, abs=1e-12
    )
    assert "not independent" in power["note"]
    assert "clustering" in power["note"]


def test_estimator_contract_excludes_trapezoidal(artifact: dict) -> None:
    """The reader's per-evidence score is a 4-5 value ladder — the exact regime
    where trapezoidal PR-AUC inflates. It must not appear as a metric here."""
    contract = artifact["estimator_contract"]
    assert "roc_auc_score" in contract["auroc"]
    assert "average_precision_score" in contract["average_precision"]
    assert "deliberately absent" in contract["trapezoidal_pr_auc"]
    assert contract["error_detection_positive_class"].endswith("INCORRECT")
    for arm_id, block in contract["sklearn_agreement"].items():
        assert block["auroc_abs_diff"] <= 1e-12, arm_id
        assert block["average_precision_abs_diff"] <= 1e-12, arm_id

    drawn = [arm for arm in artifact["arms"] if "metrics" in arm]
    readers = [arm for arm in drawn if arm["kind"] == "reader"]
    assert readers
    for arm in readers:
        assert arm["metrics"]["distinct_scores"] <= 8, (
            f"{arm['id']}: the reader's per-evidence score is expected to be a short "
            "ladder; a change here changes which estimator is safe"
        )


def test_reference_arm_is_the_strongest_sourceable_baseline(artifact: dict) -> None:
    """Deltas are quoted against a baseline, and it must be the best one.

    Quoting against a weak baseline would flatter every reader arm.
    """
    baselines = [
        arm for arm in artifact["arms"] if "metrics" in arm and arm["kind"] == "baseline"
    ]
    assert baselines
    best = max(baselines, key=lambda arm: arm["metrics"]["auroc"])
    assert artifact["reference_arm_id"] == best["id"]

    deltas = artifact["paired_delta_vs_reference"]
    assert best["id"] not in deltas, "the reference is not compared against itself"
    drawn_ids = {arm["id"] for arm in artifact["arms"] if "metrics" in arm}
    assert set(deltas) == drawn_ids - {best["id"]}
    for arm_id, block in deltas.items():
        pooled = block["pooled"]["auroc"]
        assert pooled["n_valid_resamples"] == artifact["n_bootstrap"], arm_id
        assert pooled["ci95_low"] <= pooled["mean"] <= pooled["ci95_high"], arm_id
        assert pooled["excludes_zero"] == (
            pooled["ci95_low"] > 0 or pooled["ci95_high"] < 0
        ), arm_id
        # The point delta and the bootstrap mean should agree closely; a large
        # gap means the resample distribution is not centred on the estimate.
        assert math.isclose(
            block["point"]["auroc"], pooled["mean"], abs_tol=0.01
        ), arm_id


def test_readers_are_measured_against_the_stated_expectation(artifact: dict) -> None:
    """The brief's expectation, held as a check rather than assumed.

    Source-prior baselines are constant within a source, so their per-evidence
    discrimination comes only from BETWEEN-source differences. A reader that
    actually reads the sentence should beat that. This asserts the shipped
    numbers still say what the page says they say — including the part that does
    NOT go our way, so a rerun that flips it fails loudly instead of quietly
    contradicting the prose.
    """
    by_id = {arm["id"]: arm for arm in artifact["arms"] if "metrics" in arm}
    bundled = by_id["indra-default-source-prior"]
    # An identical constant for every reader source cannot discriminate within
    # one, and there is nothing between them either: chance, by construction.
    assert bundled["constant_within_source"] is True
    assert bundled["metrics"]["auroc"] == pytest.approx(0.5, abs=0.01)
    assert bundled["metrics"]["error_detection"]["f1"] == pytest.approx(0.0, abs=1e-12)

    reference = by_id[artifact["reference_arm_id"]]
    deltas = artifact["paired_delta_vs_reference"]
    beat = {
        arm_id
        for arm_id, block in deltas.items()
        if by_id[arm_id]["kind"] == "reader"
        and block["pooled"]["auroc"]["excludes_zero"]
        and block["pooled"]["auroc"]["mean"] > 0
    }
    readers = {arm_id for arm_id, arm in by_id.items() if arm["kind"] == "reader"}
    assert beat, (
        "no reader beats the strongest source-prior baseline at per-evidence grain; "
        "that is a major finding and the page must say so, not this test pass quietly"
    )
    # Not every reader does, and the page reports that plainly.
    assert beat != readers, (
        "every reader now beats the baseline; the page's honest caveat about the "
        "smallest arm is stale and must be rewritten"
    )
    for arm_id in beat:
        assert by_id[arm_id]["metrics"]["auroc"] > reference["metrics"]["auroc"]
