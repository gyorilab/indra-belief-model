"""INDRA's installed default source priors, without a local transcription.

``noise_model.INDRA_PRIORS`` is byte-frozen with the reader implementation and
therefore cannot track INDRA's resource.  Analysis callers that claim to use
INDRA's library defaults import this module and pass ``INDRA_DEFAULT_PRIORS``
explicitly instead.

The resource is not quite a total ``source -> (rand, syst)`` mapping in INDRA
1.24: ``wormbase`` has a random-error probability but no systematic-error
probability.  It remains visible in ``declared_sources`` and
``incomplete_sources``.  Looking it up raises ``IncompleteIndraPriorError`` so
it can never disappear into the generic unknown-source fallback.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping
from importlib.resources import files
from types import MappingProxyType

INDRA_DEFAULT_PRIOR_RESOURCE = "indra/resources/default_belief_probs.json"

# ``noise_model.RECALIBRATED_PRIORS`` was written as a complete replacement for
# its old 18-row default table, not as a pure override map.  Only these five
# n>=100 rows were fitted by the benchmark; its other rows are copied defaults
# that may now be stale or absent from INDRA's installed resource.
BENCHMARK_RECALIBRATED_SOURCES = frozenset(
    {"reach", "sparser", "trips", "medscan", "rlimsp"}
)


class IncompleteIndraPriorError(ValueError):
    """A source is declared by INDRA but lacks part of its noise prior."""


class IndraPriorMapping(Mapping[str, tuple[float, float]]):
    """Read-only complete priors plus visibility for incomplete source rows.

    Iteration follows normal ``Mapping`` semantics and yields the sources that
    have both probabilities.  ``declared_sources`` is the full union of INDRA's
    ``rand`` and ``syst`` sections.  A lookup for a declared-but-incomplete
    source raises instead of making it indistinguishable from an unknown source.
    """

    def __init__(
        self,
        complete_priors: Mapping[str, tuple[float, float]],
        *,
        declared_sources: set[str] | frozenset[str],
        missing_components: Mapping[str, tuple[str, ...]],
    ) -> None:
        self._complete = MappingProxyType(dict(complete_priors))
        self._declared = frozenset(declared_sources)
        self._missing = MappingProxyType(dict(missing_components))

    def __getitem__(self, source: str) -> tuple[float, float]:
        key = source.lower()
        try:
            return self._complete[key]
        except KeyError:
            missing = self._missing.get(key)
            if missing:
                components = " and ".join(missing)
                raise IncompleteIndraPriorError(
                    f"INDRA declares source {key!r} in "
                    f"{INDRA_DEFAULT_PRIOR_RESOURCE} but omits its {components} "
                    "probability; refusing to use the unknown-source fallback"
                ) from None
            raise

    def __iter__(self) -> Iterator[str]:
        return iter(self._complete)

    def __len__(self) -> int:
        return len(self._complete)

    @property
    def complete_priors(self) -> Mapping[str, tuple[float, float]]:
        """The resource rows containing both ``rand`` and ``syst``."""
        return self._complete

    @property
    def declared_sources(self) -> frozenset[str]:
        """Every source named in either section of INDRA's resource."""
        return self._declared

    @property
    def incomplete_sources(self) -> frozenset[str]:
        """Declared sources that cannot form a ``(rand, syst)`` tuple."""
        return frozenset(self._missing)

    @property
    def missing_components(self) -> Mapping[str, tuple[str, ...]]:
        """Missing resource fields by incomplete source."""
        return self._missing

    def with_overrides(
        self,
        overrides: Mapping[str, tuple[float, float]],
    ) -> "IndraPriorMapping":
        """Return INDRA defaults with complete source-specific overrides.

        This is used for benchmark-recalibrated sources.  Sources not present in
        ``overrides`` retain their actual installed INDRA defaults rather than
        falling through to ``noise_model``'s generic fallback.
        """
        normalized = dict(self._complete)
        declared = set(self._declared)
        missing = dict(self._missing)
        for source, pair in overrides.items():
            key = source.lower()
            normalized[key] = _validated_pair(key, pair)
            declared.add(key)
            missing.pop(key, None)
        return IndraPriorMapping(
            normalized,
            declared_sources=declared,
            missing_components=missing,
        )


def _probability(section: str, source: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{INDRA_DEFAULT_PRIOR_RESOURCE}: {section}.{source} must be numeric"
        )
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"{INDRA_DEFAULT_PRIOR_RESOURCE}: {section}.{source}={value!r} "
            "must be finite and in [0, 1]"
        )
    return probability


def _validated_pair(source: str, pair: object) -> tuple[float, float]:
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise ValueError(f"override for {source!r} must be a (rand, syst) pair")
    rand = _probability("rand", source, pair[0])
    syst = _probability("syst", source, pair[1])
    if rand + syst > 1.0:
        raise ValueError(f"override for {source!r} has rand + syst > 1")
    return rand, syst


def _normalized_section(name: str, value: object) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise ValueError(
            f"{INDRA_DEFAULT_PRIOR_RESOURCE}: {name!r} must be a non-empty object"
        )
    normalized: dict[str, float] = {}
    for source, raw_probability in value.items():
        if not isinstance(source, str) or not source:
            raise ValueError(
                f"{INDRA_DEFAULT_PRIOR_RESOURCE}: {name} source names must be non-empty strings"
            )
        key = source.lower()
        if key in normalized:
            raise ValueError(
                f"{INDRA_DEFAULT_PRIOR_RESOURCE}: duplicate normalized source {key!r}"
            )
        normalized[key] = _probability(name, key, raw_probability)
    return normalized


def _load_indra_default_priors() -> tuple[IndraPriorMapping, str]:
    resource = files("indra").joinpath("resources", "default_belief_probs.json")
    raw = resource.read_bytes()
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not parse {INDRA_DEFAULT_PRIOR_RESOURCE}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{INDRA_DEFAULT_PRIOR_RESOURCE}: top level must be an object")

    rand = _normalized_section("rand", payload.get("rand"))
    syst = _normalized_section("syst", payload.get("syst"))
    declared = set(rand) | set(syst)
    complete_sources = set(rand) & set(syst)
    complete = {
        source: _validated_pair(source, (rand[source], syst[source]))
        for source in sorted(complete_sources)
    }
    missing = {
        source: tuple(
            component
            for component, section in (("rand", rand), ("syst", syst))
            if source not in section
        )
        for source in sorted(declared - complete_sources)
    }
    return (
        IndraPriorMapping(
            complete,
            declared_sources=declared,
            missing_components=missing,
        ),
        hashlib.sha256(raw).hexdigest(),
    )


INDRA_DEFAULT_PRIORS, INDRA_DEFAULT_PRIORS_SHA256 = _load_indra_default_priors()


def with_benchmark_recalibration(
    recalibrated_priors: Mapping[str, tuple[float, float]],
) -> IndraPriorMapping:
    """Layer only genuinely fitted benchmark rows onto installed defaults.

    The frozen recalibration table includes copied defaults for compatibility
    with its original callers.  Selecting the documented fitted rows here keeps
    those copies from overwriting newer INDRA defaults or resurrecting sources
    no longer declared by INDRA.
    """
    missing = BENCHMARK_RECALIBRATED_SOURCES - set(recalibrated_priors)
    if missing:
        raise ValueError(
            "recalibrated priors omit fitted benchmark sources: "
            + ", ".join(sorted(missing))
        )
    overrides = {
        source: recalibrated_priors[source]
        for source in BENCHMARK_RECALIBRATED_SOURCES
    }
    return INDRA_DEFAULT_PRIORS.with_overrides(overrides)
