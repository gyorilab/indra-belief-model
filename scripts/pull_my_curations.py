"""Pull every curation a single curator has submitted to the INDRA DB.

The keyed list-all (`GET /curation/list?api_key=`) returns the entire curation
universe (~19k rows), each carrying a `curator` email. This script pulls it once
and filters to one curator, defaulting to $INDRA_CURATOR_EMAIL.

    PYTHONPATH=src .venv/bin/python scripts/pull_my_curations.py [email]

Writes data/results/my_curations.jsonl (one raw DB row per line) and prints a
per-tag tally.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_env() -> tuple[str, str, str]:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return (
        os.environ["INDRA_DB_REST_URL"].rstrip("/"),
        os.environ["INDRA_DB_REST_API_KEY"],
        os.environ.get("INDRA_CURATOR_EMAIL", ""),
    )


def _resolve(host: str, tries: int = 20) -> str | None:
    for _ in range(tries):
        try:
            return socket.gethostbyname(host)
        except OSError:
            time.sleep(1.0)
    return None


def pull_all(url: str, key: str) -> list[dict]:
    import httpx

    host = url.split("://", 1)[1]
    ip = _resolve(host)
    if ip is None:
        raise SystemExit(f"DNS: could not resolve {host} after retries")
    print(f"resolved {host} -> {ip}", flush=True)
    last = None
    for attempt in range(20):
        try:
            with httpx.Client(
                base_url=url,
                headers={"User-Agent": "indra-belief-pull-mine/1"},
                timeout=httpx.Timeout(240.0, connect=30.0),
            ) as c:
                r = c.get("/curation/list", params={"api_key": key})
                r.raise_for_status()
                data = r.json()
                if isinstance(data, dict):
                    raise SystemExit(f"list-all returned error dict: {data}")
                return data
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"  attempt {attempt} failed: {type(e).__name__}: {e}", flush=True)
            time.sleep(min(2 * (attempt + 1), 15))
    raise SystemExit(f"list-all failed all retries: {last}")


def main() -> None:
    url, key, default_email = _load_env()
    email = (sys.argv[1] if len(sys.argv) > 1 else default_email).strip()
    if not email:
        raise SystemExit("no curator email (pass an arg or set INDRA_CURATOR_EMAIL)")

    data = pull_all(url, key)
    print(f"TOTAL curations in DB: {len(data)}", flush=True)

    mine = [d for d in data if (d.get("curator") or "").strip().lower() == email.lower()]
    print(f"\ncurations by {email!r}: {len(mine)}", flush=True)

    by_tag = Counter(d.get("tag", "") for d in mine)
    print("\n=== tag tally ===", flush=True)
    for tag, n in by_tag.most_common():
        print(f"  {tag!r:20s} {n}", flush=True)

    n_stmts = len({d.get("pa_hash") for d in mine if d.get("pa_hash") is not None})
    n_pairs = len(
        {(d.get("pa_hash"), d.get("source_hash")) for d in mine if d.get("pa_hash") is not None}
    )
    print(f"\nunique statements: {n_stmts}   unique (stmt, evidence) pairs: {n_pairs}", flush=True)

    out = ROOT / "data" / "results" / "my_curations.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for d in sorted(mine, key=lambda r: str(r.get("date", ""))):
            f.write(json.dumps(d) + "\n")
    print(f"\nSaved {len(mine)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
