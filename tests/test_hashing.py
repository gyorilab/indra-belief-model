"""Regression pins for the canonical content-address spine.

These lock the byte output of the shared codec, a known statement-id digest,
the structural sharing between assemble<->metrics, and the deliberate split
between the content-address codec (ensure_ascii=False) and the on-disk
spend-ledger encoder (ensure_ascii=True). Nothing here may be "unified".
"""

from indra_belief import hashing
from indra_belief.comparison import assemble, metrics
from indra_belief.comparison.contracts import (
    canonical_json_bytes as contracts_canonical_json_bytes,
)
from indra_belief import spend_guard


def test_canonical_codec_is_byte_stable_and_key_order_invariant():
    # Sorted keys, tight separators, raw UTF-8 (non-ASCII 'µ' -> 0xC2 0xB5).
    assert (
        hashing.canonical_json_bytes({"z": "µ", "a": {"b": 2, "a": 1}})
        == b'{"a":{"a":1,"b":2},"z":"\xc2\xb5"}'
    )
    # Key order in the input never changes the bytes.
    assert hashing.canonical_json_bytes({"a": 1, "b": 2}) == hashing.canonical_json_bytes(
        {"b": 2, "a": 1}
    )
    # Idempotent across a round trip.
    once = hashing.canonical_json_bytes({"a": 1, "b": 2})
    assert once == hashing.canonical_json_bytes({"a": 1, "b": 2})
    assert hashing.canonical_json_line({"a": 1}) == hashing.canonical_json_bytes({"a": 1}) + b"\n"
    assert (
        hashing.canonical_sha256({"a": 1})
        == hashing.sha256_bytes(hashing.canonical_json_bytes({"a": 1}))
    )


def test_contracts_reexports_the_canonical_codec():
    # The public contracts name resolves to the single canonical home.
    assert contracts_canonical_json_bytes is hashing.canonical_json_bytes


def test_ordered_statement_id_sha256_known_digest():
    # Frozen shipped digest — must never move.
    assert (
        hashing.ordered_statement_id_sha256(["a", "b"])
        == "5952341af1c0aa74032adf94b43af59d88da56610c3521e3adac30b2ef13ebc7"
    )


def test_assemble_and_metrics_share_one_implementation():
    # Structural guarantee of the producer<->consumer digest contract: the
    # producer (assemble) and consumer (metrics) are the SAME object.
    assert assemble.ordered_statement_id_sha256 is metrics.ordered_statement_id_sha256
    assert assemble.ordered_statement_id_sha256 is hashing.ordered_statement_id_sha256
    assert assemble.sha256_file is metrics.sha256_file is hashing.sha256_file
    # And the shared implementation produces the frozen digest for a DIFFERENT
    # input than the known-digest test above — an INDEPENDENT literal, so a broken
    # body fails here rather than f(x)==f(x) passing tautologically (both aliases
    # are the same object, proven above).
    assert (
        assemble.ordered_statement_id_sha256(["s1", "s2"])
        == "e6299d875df24e57807df0784695a38007645aa9a89917a3f59bfa4af75b256e"
    )


def test_ledger_encoder_is_a_deliberately_separate_frozen_codec():
    # ensure_ascii=True: non-ASCII 'α' is escaped to α (frozen ledger bytes).
    assert spend_guard._ledger_json_bytes({"x": "α"}) == b'{"x":"\\u03b1"}'
    # ensure_ascii=False content-address codec keeps raw UTF-8 for the same input.
    assert hashing.canonical_json_bytes({"x": "α"}) == b'{"x":"\xce\xb1"}'
    # The two codecs are deliberately NOT the same bytes and must not be merged.
    assert spend_guard._ledger_json_bytes({"x": "α"}) != hashing.canonical_json_bytes(
        {"x": "α"}
    )
