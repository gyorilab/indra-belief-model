r"""Pin the README's two load-bearing code claims to the code itself.

WHY THIS FILE EXISTS. README.md is read by no guard. `scripts/check_doc_anchors.py`
globs `research/*.md` only, so nothing in CI has ever looked at the README — and it
drifted: it named a variant that had not been the default since the reasoning-first
prompt shipped, and it said nothing at all about the local MLX route.

Extending that guard's glob to the README would not fix this even if it were free.
The guard's own docstring records two limits that both bite here. Tier A: a BARE
snake_case token is below its grammar, and widening the grammar was measured and
rejected at ~50% precision — so a variant NAME is invisible to it by construction.
Tier B: every path in a sentence can resolve while the sentence is false, which is
exactly the failure we had — `src/indra_belief/scorers/monolithic/scorer.py` existed
the whole time the README misdescribed its default. The guard says a human re-reading
the doc is what catches Tier B. These two tests are the narrow mechanical slice of
that job which does NOT need a human, so it should not depend on one.

EXPECTATIONS ARE DERIVED, NEVER SPELLED. Nothing here hardcodes a variant name, a
model id or a port. Each is read at runtime from `scorer.DEFAULT_VARIANT_NAME`, from
`LOCAL_MODELS["local-gemma-4-26b"]`, or from `scripts/serve_mlx.sh`. A test that
restated the values would just be a third place for the same rot to grow — the same
reason `check_doc_anchors.py` replaced its literal three-element DOCS list with a
glob.

WHY THE SECOND PATTERN IS SCOPED TO ITS LINE. `\(default (?P<v>...)\)` is the
source-tree map's spelling — `scorer.py  # MONO_VARIANT dispatch (default X)`. Left
unanchored it reaches ANY parenthesised default anywhere in the README: appending a
line like ``(default 4096)`` about some unrelated flag makes it report `4096` as a
stated variant and reds this file, which is reproduced as arm two of
`test_default_variant_anchor_is_load_bearing`. That failure mode is worse than no
test at all — a pin that breaks on edits it has no business reading gets deleted by
whoever hits it, taking the real claim with it. So the match is restricted to lines
that also name `MONO_VARIANT`, the env var `scorer.py` actually reads. Narrowing it
this way must NOT be traded for the >=1 floor below: a pattern that quietly matches
nothing passes vacuously, and that vacuum is the original defect. Both properties
are asserted, not assumed — do not "simplify" the anchor back out.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_README = _ROOT / "README.md"
_SERVE_MLX = _ROOT / "scripts" / "serve_mlx.sh"

_REGISTRY_KEY = "local-gemma-4-26b"

# Both spellings the README uses to state the default, each with the substring that
# must appear on the SAME line for the match to count (None = search the whole text).
# Prose: the sentence in "Two-tier monolithic path". Map: the aligned comment in the
# source-tree listing, which is line-scoped for the reason in the module docstring.
_DEFAULT_VARIANT_PATTERNS = (
    (r"default variant is `(?P<v>[A-Za-z0-9_]+)`", None),
    (r"\(default (?P<v>[A-Za-z0-9_]+)\)", "MONO_VARIANT"),
)


def _stated_defaults(text: str) -> list[str]:
    """Every variant name `text` states as the MONO_VARIANT default.

    Takes the text rather than reading README.md so a negative control can exercise
    the matching on synthetic input without touching the file on disk.

    Raises AssertionError if either pattern matches nothing: that >=1 floor is what
    stops a reword that simply deletes the claim from leaving the pin vacuously
    green.
    """
    stated: list[str] = []
    for pattern, anchor in _DEFAULT_VARIANT_PATTERNS:
        if anchor is None:
            chunks = [text]
        else:
            chunks = [line for line in text.splitlines() if anchor in line]
        found = [m.group("v") for chunk in chunks for m in re.finditer(pattern, chunk)]
        where = "README text" if anchor is None else f"README line naming {anchor!r}"
        assert found, (
            f"no {where} matches {pattern!r}. The default-variant claim was "
            "reworded or deleted; this pin cannot certify a sentence that is not "
            "there. Restate the default in a form this pattern reaches, or update "
            "the pattern deliberately."
        )
        stated.extend(found)
    return stated


def test_readme_states_the_actual_default_variant():
    """The README named the wrong MONO_VARIANT default and no test noticed.

    The code-side assertion (the default resolves to the reasoning-first variant)
    already lives in tests/test_monolithic_variant_profile.py and is not repeated
    here. This asserts only the DOC side: every place the README states a default
    states the one the scorer actually resolves to.
    """
    from indra_belief.scorers.monolithic import scorer as S

    for value in _stated_defaults(_README.read_text()):
        assert value == S.DEFAULT_VARIANT_NAME, (
            f"README states the default variant is {value!r}, but "
            f"scorer.DEFAULT_VARIANT_NAME is {S.DEFAULT_VARIANT_NAME!r}. The "
            "code is the truth; fix the README."
        )


def test_default_variant_anchor_is_load_bearing():
    """Both halves of the matching rule — the line anchor and the >=1 floor — bite.

    Arm two is the demonstration: delete the `"MONO_VARIANT" in line` scoping and an
    unrelated `(default N)` sentence anywhere in the README starts being read as a
    variant claim, reddening a test that has nothing to do with the edit. Arm three
    is the counterweight: narrowing must not create a pattern that passes when the
    pinned sentence is gone.
    """
    from indra_belief.scorers.monolithic import scorer as S

    text = _README.read_text()

    # (a) The README as it stands states exactly one thing: the scorer's default.
    assert set(_stated_defaults(text)) == {S.DEFAULT_VARIANT_NAME}

    # (b) An unrelated parenthesised default elsewhere in the doc is not a variant
    #     claim, and must not change what this file reads out of the README.
    noise = f"{text}\n- `--max-tokens` bounds the generation (default 4096).\n"
    assert _stated_defaults(noise) == _stated_defaults(text)
    assert "4096" not in "".join(_stated_defaults(noise))

    # (c) Strip the line the second pattern pins — derived, not spelled — and the
    #     floor must fire rather than the pattern going quietly empty.
    map_pattern, map_anchor = _DEFAULT_VARIANT_PATTERNS[1]
    stripped = "\n".join(
        line
        for line in text.splitlines()
        if not (map_anchor in line and re.search(map_pattern, line))
    )
    with pytest.raises(AssertionError, match=re.escape(map_anchor)):
        _stated_defaults(stripped)


def test_readme_local_mlx_matches_the_registry_and_the_serve_script():
    """Three artifacts have to agree or the local MLX route silently 404s.

    `scripts/serve_mlx.sh` binds a model and a port; `LOCAL_MODELS` sends requests
    to a model and a port; the README tells a reader which ones. The script's own
    header warns that changing one without the other 404s every call, but that
    warning was a comment, enforced by nothing.

    check_doc_anchors.py cannot do this even for a doc it scans: it resolves paths
    and dotted symbols, and every path in a wrong sentence still resolves (its
    Tier B). Agreement between a shell default, a dict value and prose is a
    semantic claim, so it needs a test.

    Host is deliberately NOT asserted: the registry says `localhost` and the script
    binds `127.0.0.1` on purpose. Pinning that would invent a coupling that does
    not exist.
    """
    from indra_belief.model_client import LOCAL_MODELS

    entry = LOCAL_MODELS[_REGISTRY_KEY]
    registry_model_id = entry["model_id"]
    registry_port = urlsplit(entry["base_url"]).port
    assert registry_port is not None, (
        f"{_REGISTRY_KEY} base_url {entry['base_url']!r} carries no port; this test "
        "reads the port from the URL rather than spelling it."
    )

    script = _SERVE_MLX.read_text()

    def _default(var: str) -> str:
        pattern = rf'^{var}="\$\{{{var}:-(?P<value>[^}}]+)\}}"'
        match = re.search(pattern, script, re.MULTILINE)
        assert match is not None, (
            f"could not read the {var} default out of {_SERVE_MLX}. The script's "
            f'`{var}="${{{var}:-...}}"` line moved or changed shape; re-derive the '
            "pattern rather than hardcoding the value."
        )
        return match.group("value")

    script_model = _default("MODEL")
    script_port = _default("PORT")

    assert script_model == registry_model_id, (
        f"serve_mlx.sh serves {script_model!r} but LOCAL_MODELS[{_REGISTRY_KEY!r}] "
        f"requests {registry_model_id!r} — every call 404s."
    )
    assert int(script_port) == registry_port, (
        f"serve_mlx.sh listens on {script_port} but "
        f"LOCAL_MODELS[{_REGISTRY_KEY!r}] dials {registry_port} — nothing connects."
    )

    readme = _README.read_text()
    for needle, what in (
        ("scripts/serve_mlx.sh", "the serve script's path"),
        (registry_model_id, f"the {_REGISTRY_KEY} model_id"),
        (str(registry_port), f"the {_REGISTRY_KEY} port"),
        ("~/.venvs/mlx-serve", "the separate MLX virtualenv"),
    ):
        assert needle in readme, (
            f"README.md does not mention {what} ({needle!r}). The Local MLX section "
            "is how a reader learns this route exists at all."
        )
