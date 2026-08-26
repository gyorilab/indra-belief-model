from indra_belief.scorers._shared import _extract_json


def test_extract_json_from_bare_object():
    expected = {"statement": "X activates Y", "verdict": "correct"}

    assert _extract_json('{"statement": "X activates Y", "verdict": "correct"}') == expected, (
        "a bare JSON object must parse unchanged"
    )


def test_extract_json_from_prose_wrapped_object():
    text = (
        'Context {not JSON}. Final answer: '
        '{"statement": "X activates Y", "verdict": "correct"}.'
    )

    assert _extract_json(text) == {
        "statement": "X activates Y",
        "verdict": "correct",
    }, "prose and an earlier invalid brace block must not hide the JSON object"


def test_extract_json_from_fenced_json_block():
    text = (
        "```json\n"
        '{"statement": "X activates Y", "verdict": "correct"}\n'
        "```"
    )

    assert _extract_json(text) == {
        "statement": "X activates Y",
        "verdict": "correct",
    }, "a fenced JSON object must be recovered"


def test_extract_json_cuts_unbalanced_reasoning_before_answer():
    text = (
        'Reasoning draft: "X activates Y is still under review\n'
        "</think>\n"
        '{"statement": "X activates Y", "verdict": "correct"}'
    )

    assert _extract_json(text) == {
        "statement": "X activates Y",
        "verdict": "correct",
    }, "the </think> cut must isolate JSON after an unbalanced reasoning quote"


def test_extract_json_ignores_trailing_text():
    text = (
        '{"statement": "X activates Y", "verdict": "correct"}\n'
        "Trailing note {not JSON}."
    )

    assert _extract_json(text) == {
        "statement": "X activates Y",
        "verdict": "correct",
    }, "trailing text and a later invalid brace block must not hide the JSON object"


def test_extract_json_returns_none_without_json():
    text = "X activates Y without a structured response."

    assert _extract_json(text) is None, "text without JSON must return None"


def test_extract_json_returns_last_parsing_dict():
    text = (
        'Draft: {"statement": "X activates Y", "verdict": "incorrect"}. '
        'Final: {"statement": "X activates Y", "verdict": "correct"}.'
    )

    assert _extract_json(text) == {
        "statement": "X activates Y",
        "verdict": "correct",
    }, "the final parsing dict must win over an earlier reasoning draft"


def test_extract_json_rejects_non_dict_top_level_json():
    assert _extract_json("[1, 2, 3]") is None, (
        "a non-dict top-level JSON value must return None"
    )


def test_extract_json_tracks_escaped_quotes_while_scanning_braces():
    text = (
        'Final: {"statement": "X activates Y", '
        '"raw": "he said \\"{\\" loudly"}'
    )

    assert _extract_json(text) == {
        "statement": "X activates Y",
        "raw": 'he said "{" loudly',
    }, "escaped quotes around a brace must not break brace balancing"
