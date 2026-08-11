"""Tests for the renormalised label probability and the transport normalizer."""
import math

import pytest

from indra_belief.logprobs import (
    detokenize,
    from_response,
    label_probability,
    verdict_position,
)
from indra_belief.model_client import ModelResponse, _normalize_openai_logprobs


# ── helpers ──────────────────────────────────────────────────────────────────

class _Alt:
    def __init__(self, token, logprob):
        self.token, self.logprob = token, logprob


class _Entry:
    def __init__(self, token, logprob, top=()):
        self.token, self.logprob, self.top_logprobs = token, logprob, list(top)


class _LP:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.logprobs = _LP(content) if content is not None else None


def _toks(*pairs):
    """Build a normalized logprobs array from (token, {alt: logprob}) pairs."""
    out = []
    for token, alts in pairs:
        top = [{"token": t, "logprob": lp} for t, lp in (alts or {}).items()]
        best = max(top, key=lambda d: d["logprob"])["logprob"] if top else 0.0
        out.append({"token": token, "logprob": best, "top": top})
    return out


# ── detokenize ───────────────────────────────────────────────────────────────

def test_detokenize_handles_sentencepiece_and_bpe_space_markers():
    assert detokenize("▁correct") == " correct"
    assert detokenize("Ġincorrect") == " incorrect"
    assert detokenize("correct") == "correct"


def test_detokenize_restores_byte_bpe_control_characters():
    """Qwen-family tokens arrive with newline as 'Ċ' and tab as 'ĉ'.

    Observed live from mlx_lm.server, which returns convert_ids_to_tokens output
    verbatim: a reply began ['<think>', 'Ċ', '</think>', 'ĊĊ', '{"', 'ver', ...].
    Leaving these unmapped would corrupt the reconstructed text the verdict
    regex runs against.
    """
    assert detokenize("Ċ") == "\n"
    assert detokenize("ĊĊ") == "\n\n"
    assert detokenize("ĉ") == "\t"


def test_verdict_position_survives_a_newline_between_key_and_value():
    lp = _toks(('{"verdict":', None), ('Ċ', None), ('"', None),
               ('correct', {"correct": math.log(0.9), "incorrect": math.log(0.1)}))
    assert verdict_position(lp) == 3
    assert label_probability(lp)["p_raw"] == pytest.approx(0.9)


def test_real_mlx_wire_shape_end_to_end():
    """Regression on the exact shape observed from mlx_lm.server.

    Byte-BPE tokens, a <think> preamble in the position array, and alternatives
    that include a non-ASCII near-synonym which must be excluded from both
    label masses.
    """
    seq = [("<think>", {}), ("Ċ", {}), ("</think>", {}), ("ĊĊ", {}),
           ('{"', {}), ("ver", {}), ("dict", {}), ('":', {}), ('Ġ"', {}),
           ("correct", {"correct": 0.0, "æŃ£ç¡®": -20.25,
                        "incorrect": -21.5, "cor": -22.75, "Ġcorrect": -23.25})]
    lp = _toks(*seq)
    r = label_probability(lp)
    assert r["status"] == "ok"
    assert r["position"] == 9
    assert r["both_observed"] is True
    # 'cor' is a prefix of 'correct' so it joins A1; the CJK token joins neither.
    assert r["p_raw"] > 0.999999999
    assert r["p_incorrect_mass"] == pytest.approx(math.exp(-21.5))


# ── transport normalizer ─────────────────────────────────────────────────────

def test_normalizer_returns_none_when_no_logprobs_object():
    assert _normalize_openai_logprobs(_Choice(None)) is None


def test_normalizer_returns_empty_list_for_present_but_empty_content():
    assert _normalize_openai_logprobs(_Choice([])) == []


def test_normalizer_ignores_unsorted_entry_scalar_and_takes_argmax():
    """mlx_lm builds the entry from argpartition output, which is NOT sorted.

    The entry-level token/logprob can therefore be an arbitrary member of the
    top-k. We must recover the true argmax from the alternatives.
    """
    entry = _Entry(
        token="incorrect", logprob=math.log(0.1),          # a non-argmax member
        top=[_Alt("incorrect", math.log(0.1)), _Alt("correct", math.log(0.9))],
    )
    out = _normalize_openai_logprobs(_Choice([entry]))
    assert out[0]["token"] == "correct"
    assert out[0]["logprob"] == pytest.approx(math.log(0.9))


def test_normalizer_falls_back_to_entry_scalar_without_alternatives():
    out = _normalize_openai_logprobs(_Choice([_Entry("x", math.log(0.5))]))
    assert out[0]["token"] == "x"
    assert out[0]["logprob"] == pytest.approx(math.log(0.5))
    assert out[0]["top"] == []


def test_normalizer_tolerates_bare_empty_dict_entries():
    out = _normalize_openai_logprobs(_Choice([{}]))
    assert out[0]["token"] == "" and out[0]["logprob"] == float("-inf")


# ── verdict position ─────────────────────────────────────────────────────────

def test_verdict_position_locates_value_token():
    lp = _toks(('{"', None), ('verdict', None), ('":', None), ('▁"', None),
               ('correct', {"correct": math.log(0.8)}), ('"}', None))
    # Reconstructed text is '{"verdict": "correct"}'; the value is index 4.
    assert verdict_position(lp) == 4


def test_verdict_position_takes_the_last_key_not_a_cot_rehearsal():
    """A reasoning preamble rehearses the verdict; the answer comes last."""
    lp = _toks(('I think "verdict": "incorrect" maybe. Final: {"', None),
               ('verdict', None), ('": "', None),
               ('correct', {"correct": math.log(0.7)}))
    assert verdict_position(lp) == 3


def test_verdict_position_none_when_key_absent():
    assert verdict_position(_toks(("hello", None))) is None
    assert verdict_position([]) is None


# ── label probability ────────────────────────────────────────────────────────

def test_label_probability_renormalises_over_the_two_labels_only():
    """Mass outside the label set is discarded, not counted against either."""
    lp = _toks(('{"verdict": "', None),
               ('correct', {"correct": math.log(0.6),
                            "incorrect": math.log(0.2),
                            "maybe": math.log(0.2)}))
    r = label_probability(lp)
    assert r["status"] == "ok"
    assert r["p_raw"] == pytest.approx(0.6 / 0.8)      # not 0.6
    assert r["label_mass"] == pytest.approx(0.8)
    assert r["both_observed"] is True


def test_label_probability_flags_when_losing_label_missing_from_topk():
    lp = _toks(('{"verdict": "', None),
               ('correct', {"correct": math.log(0.95), "sure": math.log(0.01)}))
    r = label_probability(lp)
    assert r["p_raw"] == pytest.approx(1.0)
    assert r["both_observed"] is False   # p_raw is a lower bound, not a value


def test_label_probability_matches_split_token_spelling():
    """`incorrect` split as `in` + `correct` still partitions correctly."""
    lp = _toks(('{"verdict": "', None),
               ('in', {"in": math.log(0.7), "correct": math.log(0.3)}))
    r = label_probability(lp)
    assert r["p_raw"] == pytest.approx(0.3)
    assert r["both_observed"] is True


def test_label_probability_is_case_and_quote_insensitive():
    lp = _toks(('{"verdict": "', None),
               ('Correct', {'"Correct': math.log(0.5), "▁INCORRECT": math.log(0.5)}))
    r = label_probability(lp)
    assert r["p_raw"] == pytest.approx(0.5)
    assert r["both_observed"] is True


def test_label_probability_reports_no_label_mass():
    lp = _toks(('{"verdict": "', None), ('zzz', {"zzz": math.log(0.9)}))
    assert label_probability(lp)["status"] == "no_label_mass"


def test_label_probability_reports_no_position():
    assert label_probability(_toks(("hi", None)))["status"] == "no_position"


# ── three-valued status propagation ──────────────────────────────────────────

@pytest.mark.parametrize("status", ["not_requested", "unsupported", "empty"])
def test_from_response_propagates_non_ok_status_without_inventing_a_number(status):
    r = from_response(ModelResponse(
        content="", reasoning="", tokens=0, raw_text="", finish_reason="stop",
        logprobs=[] if status == "empty" else None, logprobs_status=status,
    ))
    assert r["status"] == status
    assert r["p_raw"] is None


def test_from_response_computes_when_status_ok():
    lp = _toks(('{"verdict": "', None),
               ('correct', {"correct": math.log(0.9), "incorrect": math.log(0.1)}))
    r = from_response(ModelResponse(
        content="", reasoning="", tokens=2, raw_text="", finish_reason="stop",
        logprobs=lp, logprobs_status="ok",
    ))
    assert r["status"] == "ok"
    assert r["p_raw"] == pytest.approx(0.9)


def test_openai_compat_reads_mlx_reasoning_field_alias():
    """mlx_lm.server names the CoT field `reasoning`, not `reasoning_content`.

    Reading only `reasoning_content` made a deliberating reply arrive with BOTH
    content and raw_text empty, which reads as "model said nothing" rather than
    "model was truncated mid-thought".
    """
    from indra_belief.model_client import LOCAL_MODELS, ModelClient

    class _Msg:
        content = ""
        reasoning = "thinking out loud"

    class _Usage:
        completion_tokens = 700
        prompt_tokens = 10

    class _Ch:
        message = _Msg()
        finish_reason = "length"
        logprobs = None

    class _Resp:
        choices = [_Ch()]
        usage = _Usage()

    name = "_tmp_reasoning_alias"
    LOCAL_MODELS[name] = {"base_url": "http://127.0.0.1:1/v1", "model_id": "x",
                          "reasoning_in_content": False, "max_tokens": 8,
                          "timeout": 1}
    try:
        client = ModelClient(name)
        client._client = type("C", (), {"chat": type("Ch", (), {
            "completions": type("Cc", (), {
                "create": staticmethod(lambda **kw: _Resp())})()})()})()
        out = client._call_openai_compat("s", [{"role": "user", "content": "u"}],
                                         8, 0.0, 1)
        assert out.reasoning == "thinking out loud"
        assert out.raw_text.strip() == "thinking out loud"
    finally:
        LOCAL_MODELS.pop(name, None)


def test_call_refuses_logprobs_above_temperature_zero():
    """The argmax/sample divergence is silent, so the client must refuse it."""
    from indra_belief.model_client import LOCAL_MODELS, ModelClient
    name = "_tmp_logprob_temp_guard"
    LOCAL_MODELS[name] = {
        "base_url": "http://127.0.0.1:1/v1", "model_id": "x",
        "reasoning_in_content": False, "max_tokens": 8, "timeout": 1,
        "supports_logprobs": True,
    }
    try:
        client = ModelClient(name)
        with pytest.raises(ValueError, match="temperature=0"):
            client.call(system="s", messages=[{"role": "user", "content": "u"}],
                        temperature=0.1, top_logprobs=5)
    finally:
        LOCAL_MODELS.pop(name, None)


def test_default_model_response_is_not_requested():
    """Existing call sites that never mention logprobs stay valid and honest."""
    r = ModelResponse(content="x", reasoning="", tokens=1, raw_text="x",
                      finish_reason="stop")
    assert r.logprobs is None and r.logprobs_status == "not_requested"


def test_precision_limited_is_flagged_and_mass_clamped():
    """A provider that rounds the winning logprob to 0.0 makes the partition
    mass exceed 1, which is impossible. Measured live on mlx_lm: 66/100 records,
    worst 1.0564 where `correct` read exactly 1.000000 while `incorrect` still
    held 0.0564. Flag it and clamp; the ratio must be untouched.
    """
    lp = _toks(('{"verdict": "', None),
               ('correct', {"correct": 0.0, "incorrect": math.log(0.056416)}))
    r = label_probability(lp)
    assert r["precision_limited"] is True
    assert r["label_mass"] == pytest.approx(1.0)
    assert r["p_correct_mass"] + r["p_incorrect_mass"] == pytest.approx(1.0)
    # ratio preserved exactly
    assert r["p_raw"] == pytest.approx(1.0 / (1.0 + 0.056416))


def test_ordinary_case_is_not_flagged_precision_limited():
    lp = _toks(('{"verdict": "', None),
               ('correct', {"correct": math.log(0.6), "incorrect": math.log(0.2)}))
    r = label_probability(lp)
    assert r["precision_limited"] is False
    assert r["label_mass"] == pytest.approx(0.8)
