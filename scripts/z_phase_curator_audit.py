"""Z6 — curator-note audit on eval_set_v4.

For each of the 100 records in eval_set_v4, the curator wrote a note
explaining their gold-tag decision. Some notes hedge ("technically X but
allowing"), some flag upstream reader errors that are outside the
scorer's reach ("[Xa] -> HGNC:3528"), and most are clean tag-justifications.

This audit uses keyword heuristics to bucket each note. The output
estimates a *practical accuracy ceiling*:
    ceiling = 1 - hedged_count / N

A scorer that gets every non-hedged decision right and abstains on
hedged ones would hit this ceiling.

Buckets:
  clear_correct          — tag is "correct" and note has no hedging
  clear_incorrect        — tag is non-"correct" and note unambiguously supports it
  hedged_curator         — curator hedged ("should be X but…", "technically…")
  reader_error_flagged   — note explicitly notes upstream INDRA grounding error

Output: data/results/z_phase_curator_audit.md and .jsonl
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent


HEDGE_PHRASES = (
    "should be",
    "should have been",
    "allowing",
    "but allow",
    "we'll allow",
    "we will allow",
    "technically",
    "borderline",
    "debatable",
    "arguably",
    "lenient",
    "marginal",
)
READER_ERROR_PATTERNS = (
    re.compile(r"\[[A-Za-z][\w-]*\]\s*->\s*HGNC", re.IGNORECASE),
    re.compile(r"reader\s+(error|misread|conflated|misgrounded)", re.IGNORECASE),
    re.compile(r"wrong\s+grounding", re.IGNORECASE),
    re.compile(r"misgrounded", re.IGNORECASE),
    re.compile(r"grounding\s+error", re.IGNORECASE),
)


def classify_note(tag: str, note: str | None) -> str:
    note_l = (note or "").lower()
    if any(p.search(note_l) for p in READER_ERROR_PATTERNS):
        return "reader_error_flagged"
    if any(h in note_l for h in HEDGE_PHRASES):
        return "hedged_curator"
    if tag == "correct":
        return "clear_correct"
    return "clear_incorrect"


def main():
    in_path = ROOT / "data" / "benchmark" / "eval_set_v4.jsonl"
    md_path = ROOT / "data" / "results" / "z_phase_curator_audit.md"
    jsonl_path = ROOT / "data" / "results" / "z_phase_curator_audit.jsonl"
    md_path.parent.mkdir(parents=True, exist_ok=True)

    buckets: Counter[str] = Counter()
    records = []
    examples: dict[str, list[dict]] = {b: [] for b in (
        "clear_correct", "clear_incorrect", "hedged_curator", "reader_error_flagged"
    )}

    for line in open(in_path):
        r = json.loads(line)
        bucket = classify_note(r.get("tag", "?"), r.get("curator_note"))
        buckets[bucket] += 1
        rec = {
            "source_hash": r["source_hash"],
            "tag": r["tag"],
            "stmt_type": r["stmt_type"],
            "curator_note": (r.get("curator_note") or "")[:240],
            "bucket": bucket,
        }
        records.append(rec)
        if len(examples[bucket]) < 5:
            examples[bucket].append(rec)

    n = sum(buckets.values())
    hedged = buckets["hedged_curator"]
    reader_err = buckets["reader_error_flagged"]
    ceiling = 1 - (hedged / n) if n else 0
    ceiling_with_reader = 1 - ((hedged + reader_err) / n) if n else 0

    with open(jsonl_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    with open(md_path, "w") as out:
        out.write(f"# Z6 — curator-note audit (eval_set_v4, n={n})\n\n")
        out.write("Heuristic bucketing of curator notes; output of the\n")
        out.write("`scripts/z_phase_curator_audit.py` keyword classifier.\n\n")

        out.write("## Bucket distribution\n\n")
        out.write("| bucket | n | % |\n|---|---|---|\n")
        for b in ("clear_correct", "clear_incorrect", "hedged_curator", "reader_error_flagged"):
            cnt = buckets[b]
            out.write(f"| {b} | {cnt} | {cnt/n:.0%} |\n")
        out.write("\n")

        out.write("## Practical accuracy ceiling\n\n")
        out.write(f"- Hedged curator notes: **{hedged}** "
                  f"(scorer cannot beat human ambiguity)\n")
        out.write(f"- Reader-error-flagged: **{reader_err}** "
                  f"(upstream INDRA grounding errors outside scorer's reach)\n")
        out.write(f"- Strict ceiling (hedged only): **{ceiling:.0%}**\n")
        out.write(f"- Generous ceiling (hedged + reader errors): "
                  f"**{ceiling_with_reader:.0%}**\n\n")

        for bucket, exs in examples.items():
            if not exs:
                continue
            out.write(f"## Examples — {bucket}\n\n")
            for e in exs:
                sh = str(e["source_hash"])
                out.write(f"- `{sh[:12]}…` "
                          f"({e['stmt_type']}, tag={e['tag']}): "
                          f"{e['curator_note']}\n")
            out.write("\n")

    print(f"wrote {md_path}")
    print(f"wrote {jsonl_path}")
    print(f"\n=== buckets ===")
    for b in ("clear_correct", "clear_incorrect", "hedged_curator", "reader_error_flagged"):
        print(f"  {b}: {buckets[b]}")
    print(f"\nstrict ceiling: {ceiling:.0%}")
    print(f"generous ceiling (incl reader errors): {ceiling_with_reader:.0%}")


if __name__ == "__main__":
    main()
