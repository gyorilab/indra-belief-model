#!/usr/bin/env python
"""Differential harness: the three retired verdict parsers vs the unified one.

`indra_belief.verdict` replaced three parse implementations with one. This
script is the evidence that the replacement reads the corpus the same way, and
the evidence for the ONE place it deliberately does not.

    PYTHONPATH=src .venv/bin/python scripts/replay_parser_diff.py

READ-ONLY, by construction. Every `data/comparison*/runs/*/attempts.jsonl` is
opened with mode "r", never flocked, and streamed line by line — the files run
to 2 GB and their rows carry prompt bodies, so nothing is held. Nothing under
`data/` is written. Those files are a published artifact; a mutation here would
invalidate a paper number.

TWO ARMS, because the stored corpus alone cannot answer the question.

  Part A — stored responses. Every `row_status == "scored"` row on an LLM tier,
  re-read by all four parsers. This is the population the published numbers were
  computed from, so agreement here is what "this refactor moved no result" MEANS.

  Part B — truncation mutants. The real failure mode is `finish_reason="length"`,
  and the stored corpus CANNOT contain a response the batch parser failed on:
  `comparison/replay.py::error_row` writes `raw_text: ""` on every error row, so
  a parse failure erases its own evidence. Part A is therefore
  survivorship-biased, and its silence is not proof. Part B manufactures the
  missing population by cutting stored responses at seeded random offsets, which
  is exactly what truncation does to them.

The retired implementations are COPIED IN VERBATIM below rather than imported:
after the refactor they no longer exist in the tree, and reconstructing them
from git would make this script's answer depend on which commit it ran at.
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.verdict import parse_verdict  # noqa: E402

LLM_TIERS = frozenset({"llm_comprehension", "llm_tool_use"})
SEED = 1234
# Reservoir size per file x cuts per response. 15 files x 400 x 4 = 24,000
# mutants, enough to reach the ~0.1% divergence rate the fallback branches show
# without re-reading 32 GB for a second pass.
RESERVOIR = 400
CUTS = 4
EXAMPLES_PER_CLASS = 5


# ===========================================================================
# The retired implementations, verbatim
# ===========================================================================

# --- live, baseline profile: scorers/monolithic/_prompts.py ---------------

_RETIRED_GRID = {
    ("correct", "high"): 0.95,
    ("correct", "medium"): 0.80,
    ("correct", "low"): 0.65,
    ("incorrect", "low"): 0.35,
    ("incorrect", "medium"): 0.20,
    ("incorrect", "high"): 0.05,
}

_RETIRED_JSON = re.compile(
    r'\{[^{}]*?"verdict"\s*:\s*"(correct|incorrect)"[^{}]*?"confidence"\s*:\s*"(high|medium|low)"[^{}]*?\}',
    re.IGNORECASE,
)
_RETIRED_JSON_REV = re.compile(
    r'\{[^{}]*?"confidence"\s*:\s*"(high|medium|low)"[^{}]*?"verdict"\s*:\s*"(correct|incorrect)"[^{}]*?\}',
    re.IGNORECASE,
)
_RETIRED_VERDICT_PHRASES = [
    re.compile(r'"verdict"\s*:\s*"(correct|incorrect)"', re.IGNORECASE),
    re.compile(r'(?:final\s+)?(?:verdict|decision|conclusion)[^a-z]*?:[^a-z]*?(?:["\'\*]*)(correct|incorrect)', re.IGNORECASE),
    re.compile(r'\b(?:verdict|decision|answer)\s+(?:is|should be|would be|=)\s*[:"\'\*]*\s*(correct|incorrect)', re.IGNORECASE),
]
_RETIRED_CONFIDENCE_PHRASES = [
    re.compile(r'"confidence"\s*:\s*"(high|medium|low)"', re.IGNORECASE),
    re.compile(r'confidence[^a-z]*?:[^a-z]*?(?:["\'\*]*)(high|medium|low)', re.IGNORECASE),
    re.compile(r'confidence\s+(?:is|level)?[^a-z]*?(high|medium|low)', re.IGNORECASE),
    re.compile(r'with\s+(high|medium|low)\s+confidence', re.IGNORECASE),
]


def _retired_phrase_read(text: str) -> tuple[str | None, str | None]:
    if not text:
        return None, None
    matches = _RETIRED_JSON.findall(text)
    if matches:
        v, c = matches[-1]
        return v.lower(), c.lower()
    matches = _RETIRED_JSON_REV.findall(text)
    if matches:
        c, v = matches[-1]
        return v.lower(), c.lower()
    verdict = None
    for pat in _RETIRED_VERDICT_PHRASES:
        m = pat.findall(text)
        if m:
            verdict = m[-1].lower()
            break
    if not verdict:
        return None, None
    confidence = "medium"
    for pat in _RETIRED_CONFIDENCE_PHRASES:
        m = pat.findall(text)
        if m:
            confidence = m[-1].lower()
            break
    return verdict, confidence


def _retired_live_grid(verdict: str | None, confidence: str | None) -> float:
    """The fabrication, both branches: 0.5 for an absent verdict and a 0.50
    `.get` default for an on-grid verdict with an off-grid confidence."""
    if verdict is None:
        return 0.5
    return _RETIRED_GRID.get((verdict, confidence or "medium"), 0.50)


# --- live, structured profiles: scorers/monolithic/_prompts_disconfirm.py --

_LIVE_NULLISH = {"", "none", "null", "n/a", "na", "no objection", "no support", "-"}


def _retired_norm_field(v):
    if v is None:
        return None
    s = str(v).strip()
    return None if s.lower() in _LIVE_NULLISH else s


def _retired_live_read(text: str) -> dict:
    out = {"support": None, "objection": None, "verdict": None, "confidence": None}
    if not text:
        return out
    for m in reversed(list(re.finditer(r"\{[^{}]*\"verdict\"[^{}]*\}", text, re.DOTALL))):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        out["support"] = _retired_norm_field(obj.get("support"))
        out["objection"] = _retired_norm_field(obj.get("objection"))
        v = obj.get("verdict")
        out["verdict"] = v.lower() if isinstance(v, str) else None
        c = obj.get("confidence")
        out["confidence"] = c.lower() if isinstance(c, str) else None
        if out["verdict"] in ("correct", "incorrect"):
            return out
    v, c = _retired_phrase_read(text)
    out["verdict"], out["confidence"] = v, c
    return out


def _retired_live_commit(parsed: dict) -> tuple[str | None, str | None, str]:
    v, c = parsed.get("verdict"), parsed.get("confidence")
    if v is None:
        return None, None, "parse_null"
    return v, (c or "medium"), "model"


# --- batch: comparison/replay.py ------------------------------------------

_BATCH_SCORES = dict(_RETIRED_GRID)
_BATCH_NULLISH = frozenset({"", "none", "null", "n/a", "na", "no objection", "-"})
_RETIRED_BATCH_OBJECT = re.compile(r'\{[^{}]*"verdict"[^{}]*\}', re.DOTALL)
_RETIRED_BATCH_VERDICT = (
    re.compile(r'"verdict"\s*:\s*"(correct|incorrect)"', re.I),
    re.compile(r"(?:verdict|decision|conclusion)[^a-z]*:?[^a-z]*(correct|incorrect)", re.I),
    re.compile(r"(?:verdict|decision|answer)\s+(?:is|=)\s*(correct|incorrect)", re.I),
)
_RETIRED_BATCH_CONFIDENCE = (
    re.compile(r'"confidence"\s*:\s*"(high|medium|low)"', re.I),
    re.compile(r"confidence[^a-z]*:?[^a-z]*(high|medium|low)", re.I),
    re.compile(r"with\s+(high|medium|low)\s+confidence", re.I),
)


def _retired_batch_read(text: str) -> dict:
    result = {"support": None, "objection": None, "verdict": None, "confidence": None}
    if not text:
        return result
    for match in reversed(list(_RETIRED_BATCH_OBJECT.finditer(text))):
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        for field in ("support", "objection"):
            raw = value.get(field)
            normalized = None if raw is None else str(raw).strip()
            result[field] = None if (normalized or "").lower() in _BATCH_NULLISH else normalized
        verdict, confidence = value.get("verdict"), value.get("confidence")
        result["verdict"] = verdict.lower() if isinstance(verdict, str) else None
        result["confidence"] = confidence.lower() if isinstance(confidence, str) else None
        if result["verdict"] in {"correct", "incorrect"}:
            return result
    result["verdict"] = result["confidence"] = None
    for pattern in _RETIRED_BATCH_VERDICT:
        matches = pattern.findall(text)
        if matches:
            result["verdict"] = matches[-1].lower()
            break
    if result["verdict"] is None:
        return result
    result["confidence"] = "medium"
    for pattern in _RETIRED_BATCH_CONFIDENCE:
        matches = pattern.findall(text)
        if matches:
            result["confidence"] = matches[-1].lower()
            break
    return result


def _retired_batch_grid(verdict, confidence) -> float | None:
    if verdict not in {"correct", "incorrect"}:
        return None
    return _BATCH_SCORES.get((verdict, confidence or "medium"))


# ===========================================================================
# The four readings of one text
# ===========================================================================

def _retired_live(text: str) -> tuple:
    """The production live profile: structured parse, then the grid."""
    parsed = _retired_live_read(text or "")
    verdict, confidence, _rule = _retired_live_commit(parsed)
    return verdict, confidence, _retired_live_grid(verdict, confidence)


def _retired_live_baseline(text: str) -> tuple:
    """MONO_VARIANT="" — the phrase parser with no structured step."""
    verdict, confidence = _retired_phrase_read(text or "")
    if verdict is not None:
        confidence = confidence or "medium"
    return verdict, confidence, _retired_live_grid(verdict, confidence)


def _retired_batch(text: str) -> tuple:
    parsed = _retired_batch_read(text or "")
    verdict = parsed["verdict"]
    confidence = (parsed["confidence"] or "medium") if verdict is not None else None
    return verdict, confidence, _retired_batch_grid(verdict, confidence)


def _new(text: str) -> tuple:
    parsed = parse_verdict(text)
    if parsed is None:
        return None, None, None
    return parsed.label, parsed.confidence, parsed.score


READERS = (("old_live", _retired_live), ("old_live_baseline", _retired_live_baseline),
           ("old_batch", _retired_batch), ("new", _new))

# Which pairs are compared, and what a difference would mean.
PAIRS = (
    ("old_live", "old_batch"),          # the divergence this node removes
    ("old_live", "new"),                # must be empty on (verdict, confidence)
    ("old_batch", "new"),               # inherits old_live vs old_batch
    ("old_live_baseline", "new"),       # the baseline profile's own reading
)


class Diff:
    """Counts + a handful of examples per divergence class."""

    def __init__(self, examples_cap: int = EXAMPLES_PER_CLASS) -> None:
        self.rows = 0
        self.considered = 0
        self.pair_counts: Counter[str] = Counter()
        self.classes: Counter[tuple] = Counter()
        self.examples: dict[tuple, list] = {}
        self.fabricated = 0            # old_live wrote a score the grid has no cell for
        self.fabricated_examples: list = []
        # How many examples per class to keep. The report shows five; a caller
        # that wants EVERY divergence (tests/test_replay_parser_diff.py freezes
        # them) raises it. Counts never depend on this.
        self._examples_cap = examples_cap

    def observe(self, text: str, label: str = "") -> None:
        self.considered += 1
        read = {name: fn(text) for name, fn in READERS}
        for left, right in PAIRS:
            if read[left][:2] != read[right][:2]:
                key = (f"{left}!={right}", read[left][:2], read[right][:2])
                self.pair_counts[f"{left}!={right}"] += 1
                self.classes[key] += 1
                bucket = self.examples.setdefault(key, [])
                if len(bucket) < self._examples_cap:
                    # The FULL text is kept; the report tail-slices at the print
                    # site. A parser is a function of the whole reply, so an
                    # example truncated at store time cannot be re-read.
                    bucket.append((label, text))
        live_score = read["old_live"][2]
        if live_score not in _RETIRED_GRID.values():
            self.fabricated += 1
            if len(self.fabricated_examples) < self._examples_cap:
                self.fabricated_examples.append(
                    (label, read["old_live"][:2], read["new"], text))

    def merge(self, other: "Diff") -> None:
        self.rows += other.rows
        self.considered += other.considered
        self.pair_counts.update(other.pair_counts)
        self.classes.update(other.classes)
        self.fabricated += other.fabricated
        for key, bucket in other.examples.items():
            mine = self.examples.setdefault(key, [])
            mine.extend(bucket[: max(0, self._examples_cap - len(mine))])
        self.fabricated_examples.extend(
            other.fabricated_examples[
                : max(0, self._examples_cap - len(self.fabricated_examples))]
        )

    def summary(self) -> str:
        pairs = " ".join(f"{left}!={right}={self.pair_counts.get(f'{left}!={right}', 0)}"
                         for left, right in PAIRS)
        return (f"rows={self.rows} considered={self.considered} {pairs} "
                f"off_grid_live_score={self.fabricated}")

    def totals(self) -> dict:
        """The same numbers `summary()` prints, as data.

        Every PAIRS key is present even at zero — the zeroes ARE the claim, and
        a `dict(self.pair_counts)` would drop them.
        """
        return {
            "rows": self.rows,
            "considered": self.considered,
            "pairs": {f"{left}!={right}": self.pair_counts.get(f"{left}!={right}", 0)
                      for left, right in PAIRS},
            "off_grid": self.fabricated,
        }


def _attempt_files() -> list[Path]:
    return sorted(
        path
        for pattern in ("comparison", "comparison_noreason", "comparison_verdict_only")
        for path in (ROOT / "data" / pattern / "runs").glob("*/attempts.jsonl")
    )


def _scan(path: Path, rng: random.Random, *,
          examples_cap: int = EXAMPLES_PER_CLASS) -> tuple[Diff, Diff, Counter]:
    """One pass: Part A over every scored LLM row, plus a reservoir for Part B."""
    stored, mutants = Diff(examples_cap), Diff(examples_cap)
    census: Counter[str] = Counter()
    reservoir: list[tuple[str, str]] = []
    seen = 0
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            stored.rows += 1
            status = str(row.get("row_status"))
            census[status] += 1
            if status == "error":
                error = row.get("error") or {}
                census[f"err:{error.get('type')}"] += 1
                continue
            if status != "scored" or str(row.get("tier")) not in LLM_TIERS:
                continue
            text = row.get("raw_text") or ""
            if not text:
                census["scored_llm_empty_raw_text"] += 1
                continue
            label = f"{path.parent.name}:{number}"
            stored.observe(text, label)
            # Reservoir sample for Part B, seeded and order-deterministic.
            seen += 1
            if len(reservoir) < RESERVOIR:
                reservoir.append((label, text))
            else:
                slot = rng.randrange(seen)
                if slot < RESERVOIR:
                    reservoir[slot] = (label, text)
    for label, text in reservoir:
        for cut in range(CUTS):
            offset = rng.randrange(1, len(text) + 1)
            mutants.observe(text[:offset], f"{label}#cut{cut}@{offset}")
    mutants.rows = len(reservoir) * CUTS
    return stored, mutants, census


def run(files: list[Path] | None = None, *,
        examples_cap: int = EXAMPLES_PER_CLASS) -> dict:
    """The whole scan, as values. `main()` is this plus printing.

    Returns the two totalled `Diff`s, the row census, the scanned file list and
    the per-file `(relative path, stored, mutants)` triples in scan order — so
    the report re-expresses these numbers rather than re-deriving them, and a
    test can assert on the same objects the report was printed from.
    """
    paths = list(_attempt_files() if files is None else files)
    total_stored, total_mutants = Diff(examples_cap), Diff(examples_cap)
    total_census: Counter[str] = Counter()
    per_file: list[tuple[str, Diff, Diff]] = []
    for path in paths:
        rng = random.Random(SEED)
        stored, mutants, census = _scan(path, rng, examples_cap=examples_cap)
        total_stored.merge(stored)
        total_mutants.merge(mutants)
        total_census.update(census)
        per_file.append((str(path.relative_to(ROOT)), stored, mutants))
    return {
        "files": [str(path.relative_to(ROOT)) for path in paths],
        "per_file": per_file,
        "stored": total_stored,
        "mutants": total_mutants,
        "census": total_census,
    }


def main() -> int:
    files = _attempt_files()
    if not files:
        print("no data/comparison*/runs/*/attempts.jsonl found", file=sys.stderr)
        return 1
    result = run(files)
    total_stored, total_mutants = result["stored"], result["mutants"]
    total_census = result["census"]
    print(f"# {len(files)} attempt logs\n")
    for name, stored, mutants in result["per_file"]:
        print(f"{name}")
        print(f"  A stored  {stored.summary()}")
        print(f"  B mutants {mutants.summary()}")

    for name, diff in (("PART A — stored responses", total_stored),
                       ("PART B — seeded truncation mutants", total_mutants)):
        print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
        print(f"  {diff.summary()}")
        for key, count in diff.classes.most_common():
            pair, left, right = key
            print(f"\n  [{pair}] x{count}: {left} vs {right}")
            for label, text in diff.examples[key]:
                print(f"    {label}: ...{text[-240:]!r}")
        if diff.fabricated:
            print(f"\n  [old_live wrote an off-grid score] x{diff.fabricated}")
            for label, live_pair, new_read, text in diff.fabricated_examples:
                print(f"    {label}: live={live_pair} new={new_read}")
                print(f"      ...{text[-240:]!r}")

    print(f"\n{'=' * 72}\nROW CENSUS\n{'=' * 72}")
    for key, count in sorted(total_census.items()):
        print(f"  {key} = {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
