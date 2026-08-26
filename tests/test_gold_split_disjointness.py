"""Prove the fit gold and the validation gold are disjoint.

The fit gold and the validation gold must not overlap.
``scripts/check_contamination.py`` is the obvious place to look for that proof
and cannot supply it.
``scripts/check_contamination.py::find_contamination`` folds every eval path
into ONE pooled index: its ``for path in eval_paths`` loop writes every record
into a single ``eval_norm_to_records`` map. Two golds handed to
it are UNIONED, never compared. What the script actually answers is "does a
fewshot example the model sees at inference time also appear in the eval
pool" — a different question from "is the ship gate's test set held out from
the population the calibration was fit on".

So the single invariant needs two checks. The guard supplies the fewshot one
(tests/test_contamination_guard_sources.py); this file supplies the
gold-vs-gold one:

  * ``data/benchmark/eval_curation_v1.jsonl`` — the FIT gold for most profiles;
    their ``_CONFUSION`` counts were tallied on it
    (``src/indra_belief/calibration_constants.py::_CONFUSION``).
  * ``data/benchmark/external_curator_gold_v1.jsonl`` — the independent
    32-curator VALIDATION gold.

Overlap is measured at the grain the ship gate actually joins on: the
``(matches_hash, source_hash)`` pair its ``by_pair`` index is keyed on
(``scripts/calibration_ship_gate.py::statements_for_run``).
Coarser grains are pinned too, so a real leak cannot hide behind "the pair
tuple happened to differ".

Byte identity of both golds is already pinned elsewhere — see
``tests/test_soft_belief.py::test_profile_gold_digests_match_pinned_artifacts``
against ``FIT_GOLD_SHA256`` / ``EXTERNAL_GOLD_SHA256``. This file deliberately
does NOT re-hash them; it asserts about their CONTENT instead.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse, do not reimplement. ``scripts/calibration_ship_gate.py::_ukey`` (which
# masks with that module's ``HASH_MASK``) is the exact normalizer the ship gate
# joins gold on; ``scripts/check_contamination.py::_norm`` is the exact
# sentence normalizer the contamination guard uses. Copying either into this
# file would let the definitions drift apart and quietly weaken the proof.
from calibration_ship_gate import _ukey  # noqa: E402
import check_contamination as cc  # noqa: E402

from indra_belief.calibration_constants import (  # noqa: E402
    REASONING_FIRST_PROMPT_SHA256,
    fitted_calibration_for,
)

# The bedrock reasoning-first profile fits on holdout_large_fit; its fit gold is
# whatever _PROFILE_META names rather than a repository constant.
# eval_curation_v1 remains the fit gold for the other profiles.
FIT_GOLD = "data/benchmark/holdout_large_fit.jsonl"
# eval_curation_v1 is still the fit gold for gemma_remote / medpsy_remote /
# local_gemma_mlx, and it is ALSO the only gold pair that overlaps
# external_curator_gold_v1 at all. Two tests below need an overlapping pair to
# say anything: one proves `_ukey` normalization is doing real work (against a
# disjoint pair it would pass trivially), the other pins the benign residual
# overlaps. Pointing them at the new, deliberately-disjoint fit gold would turn
# both into tautologies.
NORMALIZATION_PROBE_GOLD = "data/benchmark/eval_curation_v1.jsonl"
VAL_GOLD = "data/benchmark/external_curator_gold_v1.jsonl"


def _load(rel: str) -> list[dict]:
    """Read one JSONL gold, relative to the repository root."""
    lines = (ROOT / rel).read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _pairs(rows: list[dict]) -> set[tuple[int | None, int | None]]:
    """The ship gate's join key: normalized (matches_hash, source_hash)."""
    return {(_ukey(r["matches_hash"]), _ukey(r["source_hash"])) for r in rows}


def test_fit_and_validation_golds_are_disjoint():
    """No (statement, evidence) pair is in both golds.

    This is the invariant that makes the ship gate's reported numbers mean
    anything: the profile is FIT on eval_curation_v1 and TESTED on a set it
    has never been tallied against (thresholds are selected on the fit set
    and frozen — ``scripts/calibration_ship_gate.py::training_thresholds``,
    which is the leg that restricts selection to training statements; it
    maximizes err-F1 via that module's ``best_tau`` over ``TAU_GRID``).

    Row and pair counts are asserted alongside the intersection so that a
    gold REBUILD is caught here rather than silently re-baselined: a zero
    overlap between two sets that are no longer the sets we measured proves
    nothing.
    """
    fit = _load(FIT_GOLD)
    val = _load(VAL_GOLD)

    assert len(fit) == 4303, f"{FIT_GOLD} row count changed: {len(fit)} != 4303"
    assert len(val) == 578, f"{VAL_GOLD} row count changed: {len(val)} != 578"

    fit_pairs = _pairs(fit)
    val_pairs = _pairs(val)
    assert len(fit_pairs) == 4303, f"fit unique pairs changed: {len(fit_pairs)}"
    assert len(val_pairs) == 575, f"validation unique pairs changed: {len(val_pairs)}"

    overlap = fit_pairs & val_pairs
    assert not overlap, (
        f"{FIT_GOLD} and {VAL_GOLD} overlap — the ship-gate test set is no "
        f"longer held out. {len(overlap)} shared (matches_hash, source_hash) "
        f"pairs, e.g. {sorted(overlap)[:5]}"
    )

    # Bind the shipped fit population to the population measured above,
    # rather than leaving either a free-floating literal. ``fit_unique_pairs`` is
    # ``sum(_CONFUSION[name].values())`` in
    # ``src/indra_belief/calibration_constants.py::_named_profile``, so this
    # asserts "the population I just measured is the population _CONFUSION
    # counts were drawn from" — not merely "some number equals 1604".
    profile = fitted_calibration_for(
        "bedrock-gemma-4-26b", prompt_sha256=REASONING_FIRST_PROMPT_SHA256
    )
    assert profile is not None, "the reasoning-first bedrock profile must resolve"
    assert profile["fit_unique_pairs"] == len(fit_pairs), (
        "shipped fit population and measured fit population disagree: "
        f"{profile['fit_unique_pairs']} vs {len(fit_pairs)}"
    )
    assert profile["fit_gold"] == FIT_GOLD


def test_pair_grain_zero_is_not_a_type_artifact():
    """The zero above must be a real zero, not a str-vs-int miss.

    Measured JSON types differ across the two files:
        fit  matches_hash -> Counter({'str': 1606})
        val  matches_hash -> Counter({'int': 578})
    A naive untyped join therefore returns "no overlap" for entirely the
    wrong reason. This test makes that failure mode visible: the RAW
    statement keys share nothing, while the ``_ukey``-normalized ones share
    exactly one statement. Swap ``_ukey`` for the identity function (or any
    other lossy comparison) and this test FAILS — which is the point.
    """
    fit = _load(NORMALIZATION_PROBE_GOLD)
    val = _load(VAL_GOLD)

    raw_fit = {r["matches_hash"] for r in fit}
    raw_val = {r["matches_hash"] for r in val}
    assert not (raw_fit & raw_val), (
        "raw matches_hash values now collide across golds; this test's "
        "premise (str on one side, int on the other) has changed"
    )

    key_fit = {_ukey(r["matches_hash"]) for r in fit}
    key_val = {_ukey(r["matches_hash"]) for r in val}
    assert key_fit & key_val == {32514898637890249}, (
        "normalized statement-key overlap changed; a comparison that returns "
        "the empty set here is not measuring hash identity at all: "
        f"{sorted(key_fit & key_val)}"
    )


def test_residual_coarse_grain_overlap_is_pinned():
    """Coarser than the pair, three small overlaps exist. All are benign.

    Each is a DIFFERENT CLAIM on a shared coarse key, which is why the pair
    grain is still empty. They are pinned so that growth — the signature of a
    real leak — turns this red:

      * statement grain = 1: Activation(HIF1A -> TP53), carried by five
        distinct evidences in the fit gold (pmids 28045898, 20569445,
        24835245, 24984035, 23566959) and one entirely different evidence in
        the validation gold (pmid 31776228).
      * evidence grain = 2: source_hash 22462320452265874 is Complex(CALM,
        CCND1) in the fit gold and Complex(CALM, CDK4) in the validation
        gold; source_hash 17292472718650067777 is Activation(TNFRSF1A,
        CASP8) vs Activation(TNFRSF1A, FADD). Same sentence, different
        extracted claim — exactly the pair the reader must tell apart.
      * sentence grain = 3: the two above, plus one reused review-paper
        sentence appearing under two different pmids (fit 8978681,
        validation 22251027).
    """
    fit = _load(NORMALIZATION_PROBE_GOLD)
    val = _load(VAL_GOLD)

    stmt_overlap = {_ukey(r["matches_hash"]) for r in fit} & {
        _ukey(r["matches_hash"]) for r in val
    }
    assert len(stmt_overlap) == 1, f"statement-grain overlap grew: {stmt_overlap}"

    ev_overlap = {_ukey(r["source_hash"]) for r in fit} & {
        _ukey(r["source_hash"]) for r in val
    }
    assert ev_overlap == {22462320452265874, 17292472718650067777}, (
        f"evidence-grain overlap changed: {sorted(ev_overlap)}"
    )

    # Sentence grain uses the contamination guard's own normalizer so both
    # disjointness checks speak the same dialect. Empty strings are
    # excluded: 5 validation rows carry a blank evidence_text, so the
    # validation set has 557 distinct non-empty sentences (558 if the blank is
    # counted as one). The overlap is 3 either way.
    fit_text = {cc._norm(r.get("evidence_text") or "") for r in fit} - {""}
    val_text = {cc._norm(r.get("evidence_text") or "") for r in val} - {""}
    assert len(val_text) == 557, f"validation distinct sentences changed: {len(val_text)}"
    text_overlap = fit_text & val_text
    assert len(text_overlap) == 3, (
        f"sentence-grain overlap grew to {len(text_overlap)}; a shared "
        "sentence beyond the three pinned cases is a real leak, not a "
        "coincidence of coarse keys"
    )


def test_fit_gold_fewshot_contamination_is_bounded():
    """Neither gold contains fewshot text.

    holdout_large_fit returns ZERO fewshot hits, so this test bounds growth from
    zero on BOTH golds; any fewshot reaching either one turns it red.

    eval_curation_v1 still carries its 17 hits from CONTRASTIVE_EXAMPLES, live in
    the production prompt via scorer.py; those hits are asserted where the
    profiles fitted on eval_curation_v1 are checked.
    """
    fit_hits = cc.find_contamination(eval_paths=[ROOT / FIT_GOLD])
    assert fit_hits == [], (
        f"{FIT_GOLD} was fewshot-clean at the refit and must stay so; got "
        f"{len(fit_hits)} hits from {sorted({h['source'] for h in fit_hits})}"
    )

    val_hits = cc.find_contamination(eval_paths=[ROOT / VAL_GOLD])
    assert val_hits == [], (
        f"{VAL_GOLD} is the clean set the gate reports on and must stay "
        f"fewshot-free; got {len(val_hits)} hits"
    )
