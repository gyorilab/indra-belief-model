"""INDRA curations as gold — the single source of truth for the curation domain.

Until now this logic lived, copied, across four scripts (pull_rasmachine_curations,
curation_accuracy, build_disagreement_queue, prepare_dataset) and was re-expressed
again in the SvelteKit viewer's TypeScript. The `tag == "correct"` gold atom was
independently rewritten in a half-dozen eval scripts besides. This module is the
canonical Python home; the viewer's `viewer/src/lib/data/curation.ts` is its
deliberate cross-language twin, kept in parity by tests/test_curation_parity (a
fixture both sides must reduce identically).

Five concepts, one home:

1. The curation record — what INDRA's curation DB returns per (statement, evidence).
2. The GOLD RULE — a curation tag is gold-correct iff it is exactly "correct";
   every other tag (no_relation, wrong_relation, grounding, polarity, act_vs_amt,
   hypothesis, negative_result, entity_boundaries, mod_site, other) means the
   reader's extraction is wrong. When an evidence has several curations,
   AGGREGATE with any-incorrect-wins (one curator objection flips it).
3. The JOIN KEY — curations key on the INDRA-native int pair
   (matches_hash, source_hash). Our run exports store the statement hash as a
   STRING (`indra_matches_hash`); this module owns the str->int coercion so no
   call site re-implements it.
4. The INDEX — a JSONL of curations reduced to {key -> GoldVerdict}.
5. TRANSPORT — fetching curations from the public INDRA DB REST endpoint.

Transport (httpx/async) is imported lazily so the pure domain (1-4) has no
third-party dependency and is trivially testable.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

# ── 1. the curation record ──────────────────────────────────────────────────

#: Every curation tag the INDRA DB emits. Exactly one — "correct" — means the
#: reader's extraction is supported; the rest each name a way it is wrong.
CURATION_TAGS: tuple[str, ...] = (
    "correct",
    "no_relation",
    "wrong_relation",
    "grounding",
    "polarity",
    "act_vs_amt",
    "hypothesis",
    "negative_result",
    "entity_boundaries",
    "agent_conditions",
    "mod_site",
    "other",
)

#: The one tag that denotes a correct extraction. Everything else is "incorrect".
CORRECT_TAG = "correct"


@dataclass(frozen=True)
class Curation:
    """One human curation of a (statement, evidence) extraction.

    matches_hash / source_hash are the INDRA-native int identifiers. The DB
    payload calls the statement key `pa_hash`; our pulled file mirrors it as
    `_matches_hash`. Either is accepted by `from_dict`.
    """

    matches_hash: int
    source_hash: int
    tag: str
    curator: str = ""
    date: str = ""
    text: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Curation | None":
        mh = d.get("_matches_hash", d.get("pa_hash"))
        sh = d.get("source_hash")
        if mh is None or sh is None:
            return None
        try:
            mh_i, sh_i = int(mh), int(sh)
        except (TypeError, ValueError):
            return None
        return cls(
            matches_hash=mh_i,
            source_hash=sh_i,
            tag=str(d.get("tag", "")),
            curator=str(d.get("curator", "") or ""),
            date=str(d.get("date", "") or ""),
            text=str(d.get("text", "") or ""),
        )


# ── 2. the gold rule ────────────────────────────────────────────────────────


def is_gold_correct(tag: str | None) -> bool:
    """The gold atom: a single curation tag is correct iff it is exactly "correct".

    This is the one predicate the benchmark-eval scripts each used to rewrite as
    `record["tag"] == "correct"`. Import it instead.
    """
    return tag == CORRECT_TAG


def aggregate_gold(tags: Iterable[str]) -> str | None:
    """Aggregate one evidence's curation tags into a gold verdict.

    any-incorrect-wins: "correct" iff EVERY tag is "correct"; "incorrect" if any
    tag dissents. Returns None for an empty set (uncurated). Conservative by
    design — a lone curator objection is enough to mark the extraction wrong.
    """
    tags = list(tags)
    if not tags:
        return None
    return CORRECT_TAG if all(is_gold_correct(t) for t in tags) else "incorrect"


# ── 3. the join key ─────────────────────────────────────────────────────────


def curation_key(matches_hash, source_hash) -> tuple[int, int] | None:
    """The content-addressed join key, coercing run-export string hashes to int.

    Run exports store the statement key as a string (`indra_matches_hash`) and
    the evidence key as an int (`source_hash`); curations carry both as int.
    Returns None when either hash is missing or unparseable.
    """
    if matches_hash is None or source_hash is None:
        return None
    try:
        return (int(matches_hash), int(source_hash))
    except (TypeError, ValueError):
        return None


# ── 4. the gold index ───────────────────────────────────────────────────────


@dataclass
class GoldVerdict:
    """The aggregated human verdict for one (matches_hash, source_hash)."""

    verdict: str  # "correct" | "incorrect"
    n: int
    tags: list[str]
    curators: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class CurationIndex:
    """Curations indexed by the int join key, with derived gold verdicts.

    The viewer's CurationIndex (store.ts) mirrors this shape. `present` lets a
    caller distinguish "no curations file" from "file present but this evidence
    uncurated" without raising.
    """

    by_key: dict[tuple[int, int], list[Curation]]
    gold_by_key: dict[tuple[int, int], GoldVerdict]
    present: bool = True

    @property
    def n_statements(self) -> int:
        return len({mh for mh, _ in self.by_key})

    @property
    def n_evidences(self) -> int:
        return len(self.by_key)

    def gold_for(self, matches_hash, source_hash) -> GoldVerdict | None:
        """Look up the gold verdict for an evidence by its (possibly string)
        hashes — the canonical replacement for every ad-hoc coerce-and-lookup."""
        key = curation_key(matches_hash, source_hash)
        if key is None:
            return None
        return self.gold_by_key.get(key)

    def is_curated(self, matches_hash, source_hash) -> bool:
        return self.gold_for(matches_hash, source_hash) is not None


EMPTY_INDEX = CurationIndex(by_key={}, gold_by_key={}, present=False)


def build_index(curations: Iterable[Curation]) -> CurationIndex:
    """Reduce curations into a CurationIndex (raw groups + gold verdicts)."""
    by_key: dict[tuple[int, int], list[Curation]] = defaultdict(list)
    for c in curations:
        by_key[(c.matches_hash, c.source_hash)].append(c)

    gold_by_key: dict[tuple[int, int], GoldVerdict] = {}
    for key, curs in by_key.items():
        tags = [c.tag for c in curs]
        verdict = aggregate_gold(tags) or "incorrect"
        curators, seen = [], set()
        for c in curs:
            if c.curator and c.curator not in seen:
                seen.add(c.curator)
                curators.append(c.curator)
        notes = [c.text for c in curs if c.text and c.text.strip()]
        gold_by_key[key] = GoldVerdict(verdict=verdict, n=len(curs), tags=tags, curators=curators, notes=notes)

    return CurationIndex(by_key=dict(by_key), gold_by_key=gold_by_key, present=True)


def load_index(path: str) -> CurationIndex:
    """Load a curations JSONL into a CurationIndex. Missing file -> EMPTY_INDEX
    (present=False), so gold-aware callers degrade honestly rather than crash."""
    if not path or not os.path.exists(path):
        return EMPTY_INDEX
    curations: list[Curation] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            c = Curation.from_dict(d)
            if c is not None:
                curations.append(c)
    return build_index(curations)


# ── 5. transport (lazy: no httpx import unless you pull) ─────────────────────

INDRA_DB_REST_URL = "https://db.indra.bio"
CURATION_LIST_PATH = "/curation/list"


def curation_list_url(matches_hash: int, base_url: str = INDRA_DB_REST_URL) -> str:
    """The public, key-required curation endpoint (no auth needed for by-hash)."""
    return f"{base_url.rstrip('/')}{CURATION_LIST_PATH}/{matches_hash}"


async def fetch_curations(
    matches_hashes: Iterable[int],
    *,
    base_url: str = INDRA_DB_REST_URL,
    concurrency: int = 16,
    retries: int = 4,
    backoff: float = 0.5,
    timeout: float = 30.0,
    on_progress=None,
) -> tuple[list[dict], list[int]]:
    """Async-pull curations for the given statement hashes from the INDRA DB.

    Returns (curation_dicts, failed_hashes). Each returned dict is the raw DB
    payload augmented with `_matches_hash`. Bounded concurrency + connection
    pooling (resolves the host once) + retries-with-backoff — the by-hash route
    is public, no API key. httpx is imported here so the pure domain above has
    no third-party dependency.
    """
    import asyncio

    import httpx

    hashes = list(matches_hashes)
    sem = asyncio.Semaphore(concurrency)
    out: list[dict] = []
    failed: list[int] = []
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)

    async def one(client: httpx.AsyncClient, mh: int) -> None:
        async with sem:
            for attempt in range(retries + 1):
                try:
                    r = await client.get(f"{CURATION_LIST_PATH}/{mh}")
                    if r.status_code == 200:
                        for c in r.json():
                            c["_matches_hash"] = mh
                            out.append(c)
                        if on_progress:
                            on_progress(mh, True)
                        return
                    if 400 <= r.status_code < 500:
                        if on_progress:
                            on_progress(mh, True)
                        return
                except (httpx.TransportError, httpx.HTTPError):
                    pass
                if attempt < retries:
                    await asyncio.sleep(backoff * (2**attempt))
            failed.append(mh)
            if on_progress:
                on_progress(mh, False)

    async with httpx.AsyncClient(
        base_url=base_url,
        limits=limits,
        timeout=httpx.Timeout(timeout, connect=timeout),
        headers={"User-Agent": "indra-belief-curation/1"},
    ) as client:
        await asyncio.gather(*(one(client, mh) for mh in hashes))

    return out, failed
