# The batch scoring image.
#
# WHY THIS INSTALLS NEITHER gilda NOR indra, despite both being declared in
# pyproject.toml. They are needed to GROUND an entity — to turn raw text into a
# ScoringRecord — which happens on the live path and when generating a prompt
# substrate. The batch path does not ground: it hydrates every prompt from
# content-addressed refs in the frozen substrate (`comparison/replay.py` reads
# `systems[digest]` / `prefixes[digest]`), and `src/indra_belief/comparison/`
# contains no reference to gilda at all, not even a lazy one.
#
# Measured, on the tree this image is built from:
#   * importing `indra_belief.comparison.cli` loads numpy, pandas, scipy and
#     sklearn, and loads NEITHER gilda NOR indra;
#   * `prepared_execution` and `verdict` — the two kernels the live and batch
#     paths share — pull ZERO additional modules and no heavy dependency;
#   * built with neither package installed, the batch suites
#     test_comparison_runner / test_verdict_parser / test_comparison_llm pass in
#     the container itself — see the `test` stage, which is the gate. (An
#     earlier host-side check that claimed to "block" the imports was a no-op:
#     it used the find_module/load_module finder API, removed in Python 3.12.
#     The container is the only honest check, which is why the gate lives here.)
# Omitting them saves ~1.44 GB: indra 187 MB installed plus a 470 MB ontology
# cache, gilda 2.5 MB installed plus 784 MB of resource files.
#
# Consequence, stated so it is not discovered the hard way: THIS IMAGE CANNOT
# BUILD A SUBSTRATE AND CANNOT SCORE RAW TEXT. It replays a substrate that
# already exists. An image for the live/grounding path is a separate build and
# needs both packages plus gilda's resources on a volume.
#
# Deps are installed explicitly rather than via `pip install .` because
# pyproject.toml declares gilda and indra as hard requirements, so resolving
# them would pull in exactly what this profile exists to omit. If they are ever
# moved to an optional extra, this can become `pip install .[batch]`.

ARG PYTHON_VERSION=3.13

FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Wheels first, in their own layer: this is the slow, rarely-invalidated step.
# Versions track pyproject.toml's floors. Two additions beyond its `dependencies`:
#   pandas   — not declared anywhere, but loaded by the batch entrypoint.
#   openai / anthropic — declared, but only as the OPTIONAL `[llm]` extra, so a
#     plain install omits them. The batch path needs `openai` to reach Bedrock at
#     all (its mantle endpoint is OpenAI-compatible), which the `test` stage
#     proves: without it, model_client.py:1098 raises ModuleNotFoundError and
#     test_two_clients_for_one_model_do_not_share_mutable_config fails.
#     `anthropic` is imported at model_client.py:1118 for a sibling transport;
#     included so a plan naming that provider does not fail at dispatch.
RUN pip install --no-cache-dir \
        "numpy>=1.24.0" \
        "scipy>=1.10.0" \
        "scikit-learn>=1.2.0" \
        "pandas>=2.0.0" \
        "openai>=1.0.0" \
        "anthropic>=0.18.0"

COPY pyproject.toml README.md ./
COPY src/ ./src/

# --no-deps: the dependency set above is the whole runtime, by construction.
RUN pip install --no-cache-dir --no-deps .


# The gate runs AT BUILD TIME, not in the shipped image. The claim this profile
# rests on — that the batch path needs neither gilda nor indra — is only worth
# anything if something checks it, and checking it here means a wrong image
# cannot be produced at all. Neither package is installed in this stage, so the
# absence under test is the real absence, not a mock.
FROM builder AS test

RUN pip install --no-cache-dir "pytest>=7.0"
COPY tests/ ./tests/
# tests/test_prepared_execution_parity.py is deliberately NOT here. It asserts
# that the LIVE and BATCH producers agree, so it drives the live one, which
# builds a ScoringRecord and therefore grounds — 14 of its cases fail in this
# stage for exactly the right reason. Its absence is the boundary of this
# profile, not a gap in it: the batch image cannot run the live path, and a
# gate that pretended otherwise would be the false claim this stage exists to
# prevent. That parity is covered by the full suite on a grounding-capable
# environment, which is where it belongs.
RUN python -m pytest -q --no-header -p no:cacheprovider \
        tests/test_comparison_runner.py \
        tests/test_verdict_parser.py \
        tests/test_comparison_llm.py \
    && touch /build/.batch-profile-verified


FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Non-root. The image writes only under /app/data, which is a mounted volume;
# the operator owns that directory on the host.
RUN useradd --create-home --uid 10001 belief

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin/indra-belief-comparison /usr/local/bin/indra-belief-comparison

WORKDIR /app
COPY --chown=belief:belief scripts/ ./scripts/
COPY --chown=belief:belief pyproject.toml ./

# Forces the `test` stage to build. Without this line Docker would skip it,
# since nothing else depends on it, and the gate above would be decorative.
COPY --from=test /build/.batch-profile-verified /app/.batch-profile-verified

# The CLI resolves its inputs relative to the working directory
# (cli.py: data/comparison/run_plan.json, data/results/..., data/benchmark/...),
# so the corpus mounts here. It is NOT baked: it is tens of GB and gitignored.
VOLUME ["/app/data"]

USER belief

# Fails fast and offline if the image is wrong, without touching data/ or a
# provider: the entry point must import, and the two shared kernels must load
# without gilda or indra present.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import indra_belief.comparison.cli, indra_belief.prepared_execution, indra_belief.verdict" || exit 1

ENTRYPOINT ["indra-belief-comparison"]
CMD ["--help"]
