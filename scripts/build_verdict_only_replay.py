"""Derive a verdict-only replay substrate from the frozen reasoning-first one.

The paid comparison runner never builds a prompt: it hydrates one from a
content-addressed store and digest-checks it (``replay.ReplayIndex.main_request``
raises ``ReplayError("hydrated main prompt digest differs")`` on any deviation).
The scoring prompt is therefore frozen INSIDE the substrate — the shipped one
carries ``"mono_variant": "disconfirm_relnature_rf"`` and few-shots that instruct
``Output JSON: {"relation_check": ..., "support": ..., "objection": ...,
"verdict": ..., "confidence": ...}``. Disabling the provider's chain-of-thought
does not touch any of that. Removing the scaffolding means a new substrate, and
the repo has no generator for one.

This is that generator, and it DERIVES rather than regenerates. The user message
is not stored as text — ``ReplayIndex._record`` rebuilds it from structured row
fields (claim, entity_context, abbreviation_lines, provenance, evidence text,
lookup refs). None of those change. So exactly four things do:

  1. the main system prompt        -> verdict-only (plain and tool variants)
  2. the few-shot message prefix   -> verdict-only, per statement type
  3. main_prompt_base_sha256       -> recomputed over the two above
  4. the relation-nature sub-call  -> DROPPED (a second deliberation step),
                                      so call_topology loses its first element

Everything else — every execution's coordinates, hashes, route, evidence,
entities, lookups, and the deterministic rejections — is carried across
unchanged. The consequence worth stating: the verdict-only run remains
row-for-row pairable with the thinking run on (stmt_i, evidence_i), because the
corpus and the routing are identical. Only the prompt differs.

    PYTHONPATH=src python scripts/build_verdict_only_replay.py
    PYTHONPATH=src python scripts/build_verdict_only_replay.py --verify-only
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.comparison.replay import (  # noqa: E402
    CALLABLE_ROUTES,
    DETERMINISTIC_ROUTES,
    ReplayIndex,
    prompt_sha256,
)
from indra_belief.hashing import canonical_json_line, canonical_sha256  # noqa: E402
from indra_belief.scorers.monolithic._prompts_verdict_only import (  # noqa: E402
    VERDICT_ONLY_SYSTEM_PROMPT,
    render_example,
)

SOURCE = ROOT / "data" / "comparison" / "grounding_replay"
TARGET = ROOT / "data" / "comparison_verdict_only" / "grounding_replay"

# Fields that exist only to serve the relation-nature sub-call or the note it
# splices into the user message. With that call gone they would be dead weight
# that still had to stay digest-consistent, so they are dropped outright.
_RELATION_FIELDS = (
    "relation_prompt_sha256",
    "relation_system_ref",
    "relation_alias_refs",
    "relation_note_insertion",
    "main_user_before_relation_note_sha256",
)


def _sha_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_rows(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(canonical_json_line(row) for row in rows)
    path.write_bytes(payload)
    return {
        "path": path.name,
        "sha256": _sha_text_bytes(payload),
        "bytes": len(payload),
        "rows": len(rows),
    }


def _sha_text_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _copy_table(name: str, descriptor: dict) -> dict:
    src = SOURCE / descriptor["path"]
    dst = TARGET / descriptor["path"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    payload = dst.read_bytes()
    out = dict(descriptor)
    out["sha256"] = _sha_text_bytes(payload)
    out["bytes"] = len(payload)
    if out["sha256"] != descriptor["sha256"] or out["bytes"] != descriptor["bytes"]:
        raise SystemExit(f"{name}: copy is not byte-identical to the source table")
    return out


def _lookup_guidance() -> str:
    """The tool-route system prompt is the plain one plus the lookup guidance
    block, exactly as ``monolithic.scorer._score_with_tools`` composes it."""
    from indra_belief.scorers.monolithic.scorer import _LOOKUP_GUIDANCE

    return _LOOKUP_GUIDANCE


def _build_prefixes(statement_types: list[str]) -> tuple[dict[str, str], list[dict]]:
    """One verdict-only few-shot prefix per statement type.

    Example SELECTION is untouched: `_select_examples` is deterministic in the
    statement type, so each type gets the same contrastive pairs the shipped
    substrate used. Only the rendering changes — no reason, no support, no
    objection, no relation_check.
    """
    from indra_belief.scorers.monolithic.scorer import _select_examples

    refs: dict[str, str] = {}
    components: dict[str, dict] = {}
    for stmt_type in statement_types:
        messages: list[dict[str, str]] = []
        for example in _select_examples(stmt_type):
            user, assistant = render_example(example)
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
        digest = canonical_sha256(messages)
        refs[stmt_type] = digest
        components.setdefault(digest, {"sha256": digest, "messages": messages})
    return refs, sorted(components.values(), key=lambda item: item["sha256"])


def build() -> None:
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    executions = _rows(SOURCE / "executions.jsonl")
    lookups = {row["lookup_key_sha256"]: str(row["formatted"]) for row in _rows(SOURCE / "lookups.jsonl")}

    plain_system = VERDICT_ONLY_SYSTEM_PROMPT
    tool_system = VERDICT_ONLY_SYSTEM_PROMPT + _lookup_guidance()
    plain_ref, tool_ref = _sha_text(plain_system), _sha_text(tool_system)
    systems = {plain_ref: plain_system, tool_ref: tool_system}

    statement_types = sorted({
        str(row["statement_type"]) for row in executions if str(row["route"]) in CALLABLE_ROUTES
    })
    prefix_refs, prefix_components = _build_prefixes(statement_types)
    prefixes = {item["sha256"]: [dict(m) for m in item["messages"]] for item in prefix_components}

    TARGET.mkdir(parents=True, exist_ok=True)
    new_rows: list[dict] = []
    routes = Counter()
    for row in executions:
        route = str(row["route"])
        routes[route] += 1
        out = {key: value for key, value in row.items() if key not in _RELATION_FIELDS}
        if route in CALLABLE_ROUTES:
            system_ref = tool_ref if route == "tool" else plain_ref
            prefix_ref = prefix_refs[str(row["statement_type"])]
            out["main_system_ref"] = system_ref
            out["main_message_prefix_ref"] = prefix_ref
            # Reproduce main_request's hydration EXACTLY, lookups block included:
            # that string is what the digest commits to.
            user, refs = ReplayIndex._record(row)
            block = [lookups[str(ref)] for ref in refs]
            if block:
                user += "\n\nEntity database lookups:\n" + "\n".join(block)
            messages = [*prefixes[prefix_ref], {"role": "user", "content": user}]
            out["main_prompt_base_sha256"] = prompt_sha256(systems[system_ref], messages)
            out["call_topology"] = ["monolithic_tool_context" if route == "tool" else "monolithic"]
        elif route in DETERMINISTIC_ROUTES:
            out["call_topology"] = []
        else:
            raise SystemExit(f"unknown route {route!r}")
        new_rows.append(out)

    tables = {
        "entities": _copy_table("entities", manifest["tables"]["entities"]),
        "lookups": _copy_table("lookups", manifest["tables"]["lookups"]),
        "executions": _write_rows(TARGET / "executions.jsonl", new_rows),
    }
    # relation_aliases is deliberately absent: no relation call consumes it, and
    # ReplayIndex.load treats a table missing from `tables` as empty.

    out_manifest = dict(manifest)
    out_manifest["tables"] = tables
    out_manifest["prompt_components"] = {
        "main_systems": sorted(
            ({"sha256": plain_ref, "text": plain_system}, {"sha256": tool_ref, "text": tool_system}),
            key=lambda item: item["sha256"],
        ),
        "main_message_prefixes": prefix_components,
    }
    contract = dict(manifest["generation_contract"])
    contract["mono_variant"] = "verdict_only"
    contract["plain_main_system_ref"] = plain_ref
    contract["tool_main_system_ref"] = tool_ref
    contract["message_prefix_refs_by_statement_type"] = dict(sorted(prefix_refs.items()))
    contract.pop("relation_system_ref", None)
    contract.pop("relation_output_transform", None)
    contract["derived_from"] = {
        "manifest_sha256": _sha_text_bytes((SOURCE / "manifest.json").read_bytes()),
        "changed": ["main system prompt", "few-shot prefixes", "main_prompt_base_sha256",
                    "relation-nature call removed"],
        "unchanged": ["executions", "routes", "evidence", "entities", "lookups", "coordinates"],
    }
    out_manifest["generation_contract"] = contract
    cardinality = dict(manifest.get("cardinality", {}))
    cardinality["executions"] = len(new_rows)
    cardinality["relation_executions"] = 0
    cardinality["tool_executions"] = routes["tool"]
    out_manifest["cardinality"] = cardinality

    (TARGET / "manifest.json").write_bytes(canonical_json_line(out_manifest))
    print(json.dumps({
        "target": str(TARGET.relative_to(ROOT)),
        "routes": dict(sorted(routes.items())),
        "statement_types": len(statement_types),
        "distinct_prefixes": len(prefix_components),
        "executions": len(new_rows),
        "manifest_bytes": (TARGET / "manifest.json").stat().st_size,
        "executions_bytes": tables["executions"]["bytes"],
    }, indent=2))


def verify() -> int:
    """Load the derived substrate through the real ReplayIndex, which re-hydrates
    and digest-checks every callable prompt and re-derives every deterministic
    rejection. If this passes, the runner will accept it."""
    from indra_belief.comparison.contracts import load_run_plan

    plan_path = ROOT / "data" / "comparison_verdict_only" / "run_plan.json"
    if not plan_path.exists():
        print("no run plan yet — build it before verifying end to end")
        return 1
    plan = load_run_plan(plan_path)
    index = ReplayIndex.load(plan, workload="unique_exact_pairs_primary")
    topo = Counter(tuple(row["call_topology"]) for row in index.executions)
    systems = {ref: len(text) for ref, text in index.systems.items()}
    print(json.dumps({
        "loaded_executions": len(index.executions),
        "call_topologies": {"+".join(k) or "(deterministic)": v for k, v in sorted(topo.items())},
        "distinct_systems": len(systems),
        "distinct_prefixes": len(index.prefixes),
        "mono_variant": index.manifest["generation_contract"]["mono_variant"],
    }, indent=2))
    if any("relation_nature" in k for k in topo):
        print("FAIL: a relation-nature call survived")
        return 1
    print("\nverdict-only substrate verified through the real loader")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        build()
    return verify()


if __name__ == "__main__":
    raise SystemExit(main())
