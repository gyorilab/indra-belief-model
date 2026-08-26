from __future__ import annotations

import logging
from types import SimpleNamespace

import gilda
import pytest

from indra_belief.data import entity


_CACHED_HELPERS = (
    entity._ground_cached,
    entity._get_names_cached,
    entity._get_desc_cached,
)


@pytest.fixture(autouse=True)
def isolate_gilda_helper_state():
    counts = entity.gilda_failure_counts()
    for helper in _CACHED_HELPERS:
        helper.cache_clear()
    try:
        yield
    finally:
        for helper in _CACHED_HELPERS:
            helper.cache_clear()
        with entity._GILDA_FAILURES_LOCK:
            entity._GILDA_FAILURES.clear()
            entity._GILDA_FAILURES.update(counts)


def _entity_warnings(caplog) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == "indra_belief.data.entity"
        and record.levelno == logging.WARNING
    ]


def _assert_one_increment(before: dict[str, int], key: str) -> None:
    expected = dict(before)
    expected[key] += 1
    assert entity.gilda_failure_counts() == expected


def test_ground_failure_is_warned_counted_and_not_cached(monkeypatch, caplog):
    calls = 0
    result = [SimpleNamespace(term=SimpleNamespace(), score=0.99)]

    def flaky_ground(_name):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient ground failure")
        return result

    monkeypatch.setattr(gilda, "ground", flaky_ground)
    before = entity.gilda_failure_counts()

    with caplog.at_level(
        logging.WARNING, logger="indra_belief.data.entity"
    ):
        assert entity._cached_ground("retry-ground") == []
        assert entity._cached_ground("retry-ground") is result

    warnings = _entity_warnings(caplog)
    assert calls == 2
    assert len(warnings) == 1
    assert warnings[0].getMessage() == (
        "gilda.ground failed for 'retry-ground'; treating as no grounding"
    )
    assert warnings[0].exc_info is not None
    _assert_one_increment(before, "ground")


def test_get_names_failure_is_warned_counted_and_not_cached(monkeypatch, caplog):
    calls = 0
    result = ["TP53", "Tumor protein p53"]

    def flaky_get_names(_db, _db_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient get_names failure")
        return result

    monkeypatch.setattr(gilda, "get_names", flaky_get_names)
    before = entity.gilda_failure_counts()

    with caplog.at_level(
        logging.WARNING, logger="indra_belief.data.entity"
    ):
        assert entity._cached_get_names("HGNC", "11998") == []
        assert entity._cached_get_names("HGNC", "11998") is result

    warnings = _entity_warnings(caplog)
    assert calls == 2
    assert len(warnings) == 1
    assert warnings[0].getMessage() == (
        "gilda.get_names failed for (HGNC, 11998); treating as no names"
    )
    assert warnings[0].exc_info is not None
    _assert_one_increment(before, "get_names")


def test_get_desc_failure_is_warned_counted_once_and_not_cached(
    monkeypatch, caplog
):
    calls = 0
    names = ["TP53", "Tumor protein p53 pseudogene"]

    def flaky_get_names(_db, _db_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient get_names failure")
        return names

    monkeypatch.setattr(gilda, "get_names", flaky_get_names)
    before = entity.gilda_failure_counts()

    with caplog.at_level(
        logging.WARNING, logger="indra_belief.data.entity"
    ):
        assert entity._cached_get_desc("HGNC", "11998") == ("", False)
        assert entity._cached_get_desc("HGNC", "11998") == (
            "Tumor protein p53 pseudogene",
            True,
        )

    warnings = _entity_warnings(caplog)
    assert calls == 2
    assert len(warnings) == 1
    assert warnings[0].getMessage() == (
        "gilda.get_names failed for (HGNC, 11998); treating as no names"
    )
    assert warnings[0].exc_info is not None
    _assert_one_increment(before, "get_names")
