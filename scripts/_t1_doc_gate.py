#!/usr/bin/env python3
"""Value gate for the measured truncation-rate section.

The first version of this gate grepped for the keywords ``Wilson`` / ``95% CI``
and a bracketed pair. A node then landed a placeholder reading
``Wilson interval [0.0, 1.0]. TODO.`` which satisfied it exactly — the stub had
been written to the gate's shape. The gate certified an empty section.

So this one asserts on values, not keywords: exactly one section, no placeholder,
at least one measured ``k / n`` rate, and at least one interval that is not the
vacuous ``[0.0, 1.0]``.
"""
from __future__ import annotations

import pathlib
import re
import sys

DOC = pathlib.Path("research/serving_architecture.md")
# NOTE the [^\n]* in the heading. With re.S a plain `.*$` here is greedy across
# newlines: it swallows the rest of the file and leaves the body group EMPTY, so
# every value assertion below "fails" against a perfectly good section.
HEADING = re.compile(r"^### [^\n]*Truncation rate[^\n]*\n(.*?)(?=^### |\Z)", re.M | re.S)
RATE = re.compile(r"\b\d+\s*/\s*\d+\b")
INTERVAL = re.compile(r"\[\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\s*\]")


def main() -> int:
    sections = HEADING.findall(DOC.read_text(encoding="utf-8"))
    if len(sections) != 1:
        print(f"FAIL: expected exactly 1 Truncation-rate section, found {len(sections)}")
        return 1
    body = sections[0]
    if "TODO" in body:
        print("FAIL: Truncation-rate section still carries a TODO placeholder")
        return 1
    if not RATE.search(body):
        print("FAIL: no measured k / n rate in the section")
        return 1
    intervals = INTERVAL.findall(body)
    if not intervals:
        print("FAIL: no interval in the section")
        return 1
    if all(float(lo) == 0.0 and float(hi) == 1.0 for lo, hi in intervals):
        print("FAIL: only vacuous [0.0, 1.0] intervals — that is a placeholder, not a measurement")
        return 1
    print(f"T1 doc OK: {len(intervals)} interval(s), {len(RATE.findall(body))} rate(s), no placeholder")
    return 0


if __name__ == "__main__":
    sys.exit(main())
