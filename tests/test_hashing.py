"""Regression pins for the canonical content-address spine.

These lock the byte output of the shared codec and a known statement-id digest.

Three further pins stood here: that `comparison.contracts` re-exported this
codec, that `assemble` and `metrics` shared one implementation, and that the
on-disk spend-ledger encoder (ensure_ascii=True) was a deliberately SEPARATE
codec from this one (ensure_ascii=False). All three named modules have been
removed, so the sharing they pinned no longer has two ends. What survives is
the codec itself, and it still may not be "unified" with anything.
"""

from indra_belief import hashing


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



def test_ordered_statement_id_sha256_known_digest():
    # Frozen shipped digest — must never move.
    assert (
        hashing.ordered_statement_id_sha256(["a", "b"])
        == "5952341af1c0aa74032adf94b43af59d88da56610c3521e3adac30b2ef13ebc7"
    )


