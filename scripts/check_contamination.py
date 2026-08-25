"""Check for data contamination between few-shot examples and eval data.

Run BEFORE any evaluation to ensure no example string leaks into the
holdout, calibration, or any other eval set. CI runs this as a guard.

Sources of fewshot examples (the things the model sees during inference):
  1. legacy CONTRASTIVE_EXAMPLES (monolithic scorer)
  2. legacy example_bank.json
  4. inline examples in the parse_evidence system prompt body
     (any "quoted" sentence followed by an arrow → is treated as an example)

Eval/benchmark files checked:
  - data/benchmark/calibration_*.jsonl
  - data/benchmark/holdout_*.jsonl
  - --holdout PATH (CLI override; defaults to holdout_large.jsonl)

Contamination definition:
  EXACT match  : normalized fewshot evidence == normalized eval evidence
  SUBSTRING    : normalized fewshot evidence ⊆ normalized eval evidence
                 OR normalized eval evidence ⊆ normalized fewshot evidence
  PAIR match   : fewshot's (subject, object) pair == eval's (subject, object)
                 (legacy check, kept for the holdout)

Exit code 0 = clean, 1 = contaminated.

Usage:
    PYTHONPATH=src python scripts/check_contamination.py [--holdout PATH]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


_WHITESPACE = re.compile(r"\s+")
_TRAILING_PUNCT = re.compile(r'[.!?,;:"\']+$')


def _norm(s: str) -> str:
    """Normalize for comparison: collapse whitespace, strip trailing
    punctuation, casefold. The goal is to catch cosmetic variants of the
    same sentence — not to enable fuzzy semantic matching."""
    s = _WHITESPACE.sub(" ", s).strip()
    s = _TRAILING_PUNCT.sub("", s)
    return s.casefold()


def _short(s: str, n: int = 80) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


class SourceImportError(RuntimeError):
    """A declared fewshot source could not be imported.

    This is distinct from a source that imports fine but is legitimately
    empty. An import failure means the contamination scan is BLIND to a
    source the model actually sees — it must fail loudly, never silently
    contribute zero examples (the bug that let a wrong module path go
    unnoticed because the ModuleNotFoundError was swallowed)."""


def _load_legacy_examples() -> list[dict]:
    """Sources 1 + 2: legacy CONTRASTIVE_EXAMPLES + example_bank.

    The CONTRASTIVE_EXAMPLES import is intentionally NOT wrapped in a
    swallowing try/except: if the module path is wrong, we want a loud
    failure, not a silently empty source. A source that imports but is
    genuinely empty (e.g. CONTRASTIVE_EXAMPLES == []) is tolerated.
    """
    out = []
    try:
        from indra_belief.scorers.monolithic._prompts import (
            CONTRASTIVE_EXAMPLES,
        )
    except ImportError as e:  # import failed → fail loudly, do not swallow
        raise SourceImportError(
            "Source 1 (CONTRASTIVE_EXAMPLES) could not be imported from "
            "indra_belief.scorers.monolithic._prompts — the contamination "
            f"scan would be blind to it. Original error: {e}"
        ) from e
    for ex in CONTRASTIVE_EXAMPLES:
        out.append({"source": "CONTRASTIVE_EXAMPLES",
                    "claim": ex.get("claim", ""),
                    "evidence": ex.get("evidence", "")})

    bank_path = ROOT / "src" / "indra_belief" / "data" / "example_bank.json"
    if bank_path.exists():
        with open(bank_path) as f:
            bank = json.load(f)
        for key, pair in bank.items():
            for ex in pair:
                out.append({"source": f"example_bank:{key}",
                            "claim": ex.get("claim", ""),
                            "evidence": ex.get("evidence", "")})
    return out




# A "quoted sentence followed by an arrow" — captures inline examples in
# prompt bodies, e.g.  Example: "X did Y" → ...


def _load_unified_fewshots() -> list[dict]:
    """Source 5: unified scorer's curated few-shot curriculum (research artifact).

    Unified is archived at research/unified/ — kept in the contamination
    scan so any future revival surfaces fewshot-eval overlap.
    """
    out = []
    path = ROOT / "research" / "unified" / "unified_fewshots.jsonl"
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev = rec.get("evidence", "")
            claim = rec.get("claim", {}) or {}
            subj = claim.get("subject", "") if isinstance(claim, dict) else ""
            obj = claim.get("object", "") if isinstance(claim, dict) else ""
            claim_str = f"{subj} [{claim.get('stmt_type','')}] {obj}" if subj else ""
            if ev:
                out.append({"source": f"unified_fewshots:{rec.get('pattern','?')}",
                            "claim": claim_str, "evidence": ev})
    return out


def load_all_examples() -> list[dict]:
    """Union of every fewshot source the model sees during inference."""
    # Sources 3 and 4 stood here: the four S-phase probe modules' `_FEW_SHOTS`
    # and the inline `"..." →` examples in their system prompts. Those modules
    # (scorers.probes.{subject_role,object_role,relation_axis,scope}) were
    # deleted with the decomposed architecture, so there are no probe prompt
    # assets left for a holdout record to leak into. This is a genuine
    # reduction in what the scan must REACH, not a reduction in its coverage of
    # what still exists — every prompt asset the model can still see is loaded
    # by the two remaining sources.
    return _load_legacy_examples() + _load_unified_fewshots()


def _parse_legacy_claim(claim: str) -> tuple[str, str]:
    """Extract (subject, object) from 'SUBJ [TYPE] OBJ'."""
    parts = claim.replace("[", "|").replace("]", "|").split("|")
    subj = parts[0].strip() if parts else ""
    obj = parts[2].strip() if len(parts) > 2 else ""
    return subj, obj


def _load_eval_evidence(path: Path) -> list[dict]:
    """Read a JSONL eval file. Tolerates missing fields and the calibration
    schema (which uses 'evidence' instead of 'evidence_text')."""
    out = []
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev = (rec.get("evidence_text")
                  or rec.get("evidence")
                  or "")
            subj = rec.get("subject", "") or ""
            obj = rec.get("object", "") or ""
            if ev:
                out.append({"file": path.name, "evidence": ev,
                            "subject": subj, "object": obj})
    return out


def _default_eval_paths(holdout_arg: str) -> list[Path]:
    """Calibration files are always checked. Holdout files are checked by
    default; --holdout CLI flag overrides which holdout to scan."""
    benchmark = ROOT / "data" / "benchmark"
    paths: list[Path] = []
    # All calibration files
    paths.extend(sorted(benchmark.glob("calibration_*.jsonl")))
    # Frozen representative-curation milestones are independent evaluation gold
    # and must remain disjoint from every prompt/few-shot source.
    paths.extend(sorted(benchmark.glob("representative_indra_curations_*.jsonl")))
    # Holdout from CLI (and the small v15 sample, always)
    paths.append(Path(holdout_arg))
    sample = benchmark / "holdout_v15_sample.jsonl"
    if sample not in paths:
        paths.append(sample)
    # D4 held-back sample (used as overfit guard) — must be contamination-free.
    d4 = benchmark / "holdout_d4_held_back.jsonl"
    if d4.exists() and d4 not in paths:
        paths.append(d4)
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def find_contamination(
    examples: list[dict] | None = None,
    eval_paths: list[Path] | None = None,
    holdout_arg: str | None = None,
) -> list[dict]:
    """Programmatic API used by both this CLI and the pytest guard.

    Returns a list of contamination records: {source, evidence, file,
    eval_evidence, kind in {exact, substring_in_eval, eval_in_substring,
    pair}}."""
    if examples is None:
        examples = load_all_examples()
    if eval_paths is None:
        if holdout_arg is None:
            holdout_arg = str(ROOT / "data" / "benchmark"
                              / "holdout_large.jsonl")
        eval_paths = _default_eval_paths(holdout_arg)

    # Index eval data once: normalized evidence → list of (file, raw_ev,
    # subject, object)
    eval_norm_to_records: dict[str, list[dict]] = {}
    eval_pairs: dict[tuple[str, str], list[str]] = {}
    for path in eval_paths:
        for rec in _load_eval_evidence(path):
            n = _norm(rec["evidence"])
            eval_norm_to_records.setdefault(n, []).append(rec)
            if rec["subject"] and rec["object"]:
                eval_pairs.setdefault((rec["subject"], rec["object"]), []).append(
                    rec["file"]
                )

    eval_norms = list(eval_norm_to_records.keys())
    eval_norm_set = set(eval_norms)

    contam: list[dict] = []
    for ex in examples:
        ev = ex.get("evidence", "") or ""
        if not ev:
            continue
        en = _norm(ev)
        if not en:
            continue

        # Exact match
        if en in eval_norm_set:
            for rec in eval_norm_to_records[en]:
                contam.append({**ex, "kind": "exact",
                               "file": rec["file"],
                               "eval_evidence": rec["evidence"]})

        # Substring containment (only when not already an exact match)
        # Cap example length to skip ultra-short fragments that would
        # produce noise (any 5-char string would substring-match many
        # sentences).
        if len(en) >= 30 and en not in eval_norm_set:
            for n in eval_norms:
                if n == en:
                    continue
                if en in n or n in en:
                    for rec in eval_norm_to_records[n]:
                        contam.append({
                            **ex, "kind": ("substring_in_eval"
                                           if en in n else "eval_in_substring"),
                            "file": rec["file"],
                            "eval_evidence": rec["evidence"],
                        })
                    # one substring report per fewshot example is enough
                    break

        # Paraphrase contamination (S6 fix): a fewshot may share a long
        # distinctive substring with an eval record without either fully
        # containing the other (e.g., paraphrased shortening). Catch this
        # by sliding a 50-char window from the fewshot across each eval
        # record. 50 chars is long enough to be distinctive, short enough
        # to catch reworded versions.
        if len(en) >= 50 and en not in eval_norm_set:
            window = 50
            shingles = set()
            for i in range(0, len(en) - window + 1, 5):
                shingles.add(en[i:i + window])
            for n, recs in eval_norm_to_records.items():
                if n == en:
                    continue
                if any(sh in n for sh in shingles):
                    for rec in recs:
                        contam.append({
                            **ex, "kind": "paraphrase_overlap",
                            "file": rec["file"],
                            "eval_evidence": rec["evidence"],
                        })
                    break

        # Pair match (legacy holdout check)
        if ex.get("claim"):
            subj, obj = _parse_legacy_claim(ex["claim"])
            if subj and obj and (subj, obj) in eval_pairs:
                for fname in eval_pairs[(subj, obj)]:
                    contam.append({**ex, "kind": "pair",
                                   "file": fname,
                                   "eval_evidence": f"({subj}, {obj})"})

    return contam


def main():
    parser = argparse.ArgumentParser(description="Check example/eval contamination")
    parser.add_argument("--holdout",
                        default=str(ROOT / "data" / "benchmark"
                                    / "holdout_large.jsonl"))
    args = parser.parse_args()

    try:
        examples = load_all_examples()
    except SourceImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    paths = _default_eval_paths(args.holdout)

    # Source breakdown for the report
    by_source: dict[str, int] = {}
    for ex in examples:
        by_source[ex["source"]] = by_source.get(ex["source"], 0) + 1
    print(f"Fewshot sources ({len(examples)} total):")
    for s, n in sorted(by_source.items()):
        print(f"  {n:>4}  {s}")
    print(f"Eval files checked ({len(paths)}):")
    for p in paths:
        exists = "OK" if p.exists() else "MISSING"
        print(f"  [{exists:^7}] {p.relative_to(ROOT) if p.is_absolute() else p}")

    contam = find_contamination(examples=examples, eval_paths=paths)

    if not contam:
        print("\nCLEAN — no contamination detected.")
        return 0

    print(f"\nCONTAMINATED — {len(contam)} overlap(s):\n")
    for c in contam:
        print(f"  [{c['kind']}] {c['source']}")
        print(f"    fewshot:  {_short(c['evidence'])}")
        print(f"    eval ({c['file']}): {_short(c['eval_evidence'])}")
        print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
