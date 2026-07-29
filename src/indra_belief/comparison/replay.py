"""Provider-free prompt replay, raw-row validation, and append-only resume."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import stat
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .contracts import (
    Action,
    ContractError,
    FileCapture,
    FileDescriptor,
    RunPlan,
    canonical_json_line,
    canonical_sha256,
    stable_read,
    strict_json_loads,
)


HEX64 = re.compile(r"[0-9a-f]{64}")
HEX32 = re.compile(r"[0-9a-f]{32}")
CALLABLE_ROUTES = frozenset({"plain", "tool"})
DETERMINISTIC_ROUTES = frozenset(
    {"no_text", "deterministic_mismatch", "deterministic_pseudogene"}
)
VERDICT_SCORES = {
    ("correct", "high"): 0.95,
    ("correct", "medium"): 0.80,
    ("correct", "low"): 0.65,
    ("incorrect", "low"): 0.35,
    ("incorrect", "medium"): 0.20,
    ("incorrect", "high"): 0.05,
}
_NULLISH = frozenset({"", "none", "null", "n/a", "na", "no objection", "-"})
_VERDICT_OBJECT = re.compile(r'\{[^{}]*"verdict"[^{}]*\}', re.DOTALL)
_VERDICT_PATTERNS = (
    re.compile(r'"verdict"\s*:\s*"(correct|incorrect)"', re.I),
    re.compile(r"(?:verdict|decision|conclusion)[^a-z]*:?[^a-z]*(correct|incorrect)", re.I),
    re.compile(r"(?:verdict|decision|answer)\s+(?:is|=)\s*(correct|incorrect)", re.I),
)
_CONFIDENCE_PATTERNS = (
    re.compile(r'"confidence"\s*:\s*"(high|medium|low)"', re.I),
    re.compile(r"confidence[^a-z]*:?[^a-z]*(high|medium|low)", re.I),
    re.compile(r"with\s+(high|medium|low)\s+confidence", re.I),
)
_FORBIDDEN_INPUT_KEYS = frozenset({
    "belief", "correct", "curation", "curations", "curator", "curator_note",
    "gold", "incorrect", "label", "labels", "model_output", "model_outputs",
    "model_response", "model_responses", "predicted_label", "prediction",
    "predictions", "tag", "tags", "verdict", "verdicts",
})


class ReplayError(ContractError): pass


class ResponseLike(Protocol):
    content: str
    reasoning: str
    raw_text: str
    tokens: int
    prompt_tokens: int
    finish_reason: str
    reasoning_trace: dict[str, Any]


class ClientLike(Protocol):
    def call(self, *args: Any, **kwargs: Any) -> ResponseLike: ...
    def pop_call_log(self) -> list[dict[str, Any]]: ...


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prompt_sha256(system: str, messages: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256({"system": system, "messages": list(messages)})


def parse_structured(text: str) -> dict[str, str | None]:
    result: dict[str, str | None] = {
        "support": None, "objection": None, "verdict": None, "confidence": None
    }
    if not text:
        return result
    for match in reversed(list(_VERDICT_OBJECT.finditer(text))):
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        for field in ("support", "objection"):
            raw = value.get(field)
            normalized = None if raw is None else str(raw).strip()
            result[field] = None if (normalized or "").lower() in _NULLISH else normalized
        verdict, confidence = value.get("verdict"), value.get("confidence")
        result["verdict"] = verdict.lower() if isinstance(verdict, str) else None
        result["confidence"] = confidence.lower() if isinstance(confidence, str) else None
        if result["verdict"] in {"correct", "incorrect"}:
            return result
    result["verdict"] = result["confidence"] = None
    for pattern in _VERDICT_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            result["verdict"] = matches[-1].lower()
            break
    if result["verdict"] is None:
        return result
    result["confidence"] = "medium"
    for pattern in _CONFIDENCE_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            result["confidence"] = matches[-1].lower()
            break
    return result


def parse_response(response: ResponseLike) -> tuple[str | None, str | None]:
    for text in (response.content, response.raw_text):
        parsed = parse_structured(text or "")
        if parsed["verdict"] is not None:
            return parsed["verdict"], parsed["confidence"] or "medium"
    return None, None


def verdict_score(verdict: str | None, confidence: str | None) -> float:
    return VERDICT_SCORES.get((verdict, confidence or "medium"), 0.5) if verdict else 0.5


def _canonical_rows(descriptor: FileDescriptor, *, context: str) -> tuple[list[dict[str, Any]], FileCapture]:
    capture = descriptor.capture(context=context)
    if capture.payload and not capture.payload.endswith(b"\n"):
        raise ReplayError(f"{context} lacks a trailing newline")
    rows: list[dict[str, Any]] = []
    for number, raw in enumerate(capture.payload.splitlines(keepends=True), start=1):
        if raw == b"\n":
            raise ReplayError(f"{context} contains a blank row at {number}")
        value = strict_json_loads(raw[:-1], context=f"{context} row {number}")
        if not isinstance(value, dict) or canonical_json_line(value) != raw:
            raise ReplayError(f"{context} row {number} is not canonical JSON")
        rows.append(value)
    return rows, capture


def _reject_labels(value: Any, *, context: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            if normalized in _FORBIDDEN_INPUT_KEYS:
                raise ReplayError(f"{context} contains label-bearing key {key!r}")
            _reject_labels(child, context=f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_labels(child, context=f"{context}[{index}]")


def _unique_index(rows: Sequence[Mapping[str, Any]], key: str, *, context: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or value in result:
            raise ReplayError(f"{context} key is absent or repeated")
        result[value] = row
    return result


@dataclass(frozen=True)
class ReplayIndex:
    manifest: Mapping[str, Any]
    captures: tuple[FileCapture, ...]
    systems: Mapping[str, str]
    prefixes: Mapping[str, tuple[Mapping[str, str], ...]]
    entities: Mapping[str, Mapping[str, Any]]
    lookups: Mapping[str, str]
    relation_aliases: Mapping[str, Mapping[str, Any] | None]
    executions: tuple[Mapping[str, Any], ...]

    @classmethod
    def load(cls, plan: RunPlan, *, workload: str | None = None) -> "ReplayIndex":
        manifest_capture = plan.replay_manifest.capture(context="replay manifest")
        manifest = strict_json_loads(manifest_capture.payload, context="replay manifest")
        if not isinstance(manifest, dict):
            raise ReplayError("replay manifest is not an object")
        if canonical_json_line(manifest) != manifest_capture.payload:
            raise ReplayError("replay manifest is not canonical one-line JSON")
        if manifest.get("artifact_kind") != "indra_belief_grounding_replay" or manifest.get("status") != "ready":
            raise ReplayError("replay manifest is not the ready semantic grounding replay")
        tables = manifest.get("tables")
        if not isinstance(tables, Mapping) or "executions" not in tables:
            raise ReplayError("replay manifest lacks an executions table")
        loaded: dict[str, list[dict[str, Any]]] = {}
        captures = [manifest_capture]
        for name in ("entities", "lookups", "relation_aliases", "executions"):
            if name not in tables:
                loaded[name] = []
                continue
            descriptor = FileDescriptor.from_value(
                tables[name], owner=plan.replay_manifest.path, root=plan.root
            )
            rows, capture = _canonical_rows(descriptor, context=f"replay {name}")
            loaded[name], captures = rows, [*captures, capture]
        components = manifest.get("prompt_components", {})
        if not isinstance(components, Mapping):
            raise ReplayError("replay prompt_components is not an object")
        systems: dict[str, str] = {}
        for item in components.get("main_systems", []):
            if not isinstance(item, Mapping) or set(item) != {"sha256", "text"}:
                raise ReplayError("main system component is malformed")
            text, digest = item["text"], item["sha256"]
            if not isinstance(text, str) or digest != _sha_text(text) or digest in systems:
                raise ReplayError("main system component digest differs or repeats")
            systems[digest] = text
        relation = components.get("relation_system")
        if relation is not None:
            if not isinstance(relation, Mapping) or set(relation) != {"sha256", "text"}:
                raise ReplayError("relation system component is malformed")
            if relation["sha256"] != _sha_text(relation["text"]):
                raise ReplayError("relation system digest differs")
            systems[str(relation["sha256"])] = str(relation["text"])
        prefixes: dict[str, tuple[Mapping[str, str], ...]] = {}
        for item in components.get("main_message_prefixes", []):
            if not isinstance(item, Mapping) or set(item) != {"sha256", "messages"}:
                raise ReplayError("message-prefix component is malformed")
            messages = item["messages"]
            if not isinstance(messages, list) or any(
                not isinstance(row, Mapping) or set(row) != {"role", "content"}
                or not all(isinstance(row[key], str) for key in row)
                for row in messages
            ):
                raise ReplayError("message-prefix rows are malformed")
            digest = canonical_sha256(messages)
            if item["sha256"] != digest or digest in prefixes:
                raise ReplayError("message-prefix digest differs or repeats")
            prefixes[digest] = tuple(dict(row) for row in messages)
        executions = tuple(
            row for row in loaded["executions"]
            if workload is None or row.get("workload") == workload
        )
        if not executions:
            raise ReplayError(f"replay has no workload {workload!r}")
        if workload is not None:
            declared = [row for row in manifest.get("workloads", []) if row.get("name") == workload]
            if declared and (len(declared) != 1 or declared[0].get("execution_rows") != len(executions)):
                raise ReplayError("workload cardinality differs from its manifest")
        entities = _unique_index(loaded["entities"], "entity_key_sha256", context="entity")
        lookups = _unique_index(loaded["lookups"], "lookup_key_sha256", context="lookup")
        relation_aliases = _unique_index(
            loaded["relation_aliases"], "relation_alias_key_sha256", context="relation alias"
        )
        result = cls(
            manifest,
            tuple(captures),
            systems,
            prefixes,
            entities,
            {key: str(row["formatted"]) for key, row in lookups.items()},
            {key: row.get("grounding") for key, row in relation_aliases.items()},
            executions,
        )
        result._validate_all()
        return result

    def assert_current(self) -> None:
        for capture in self.captures:
            capture.assert_current()

    def _validate_all(self) -> None:
        seen: set[tuple[str, int, int]] = set()
        for number, row in enumerate(self.executions, start=1):
            _reject_labels(row, context=f"execution {number}")
            try:
                coordinates = (row["stmt_i"], row["evidence_i"])
                key = (int(coordinates[0]), int(coordinates[1]))
                route = str(row["route"])
                topology = row["call_topology"]
                hashes = (row["paper_statement_hash"], row["source_hash"], row["evidence_json_sha256"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ReplayError(f"execution {number} lacks its semantic identity") from exc
            identity_key = (str(row.get("workload")), *key)
            if identity_key in seen or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in coordinates
            ) or not all(
                isinstance(value, str) for value in hashes
            ) or HEX64.fullmatch(str(hashes[2])) is None:
                raise ReplayError(f"execution {number} identity is invalid or repeated")
            seen.add(identity_key)
            expected_key = canonical_sha256([
                row["workload"], key[0], key[1], str(hashes[0]), str(hashes[1]), hashes[2]
            ])
            if row.get("execution_key_sha256") not in {None, expected_key}:
                raise ReplayError(f"execution {number} key digest differs")
            expected_main = "monolithic_tool_context" if route == "tool" else "monolithic"
            expected_topology = (
                (["relation_nature", expected_main] if row.get("relation_prompt_sha256") else [expected_main])
                if route in CALLABLE_ROUTES else []
            )
            if route not in CALLABLE_ROUTES | DETERMINISTIC_ROUTES or topology != expected_topology:
                raise ReplayError(f"execution {number} route/call topology differs")
            if route in CALLABLE_ROUTES:
                self.main_request(row)
                if row.get("relation_prompt_sha256"):
                    self.relation_request(row)
            else:
                self.deterministic_result(row)

    def select(self, keys: tuple[Mapping[str, Any], ...] | None) -> "ReplayIndex":
        if keys is None:
            return self
        by_key = {(int(row["stmt_i"]), int(row["evidence_i"])): row for row in self.executions}
        selected: list[Mapping[str, Any]] = []
        seen: set[tuple[int, int]] = set()
        for entry in keys:
            if set(entry) not in (
                {"stmt_i", "evidence_i", "execution_sha256"},
                {"stmt_i", "evidence_i", "execution_key_sha256"},
            ):
                raise ReplayError("execution key fields differ")
            key = (entry.get("stmt_i"), entry.get("evidence_i"))
            if any(isinstance(value, bool) or not isinstance(value, int) for value in key):
                raise ReplayError("execution key coordinates are invalid")
            source = by_key.get(key)
            if source is None or key in seen:
                raise ReplayError("execution key is foreign or repeated")
            digest_field = "execution_sha256" if "execution_sha256" in entry else "execution_key_sha256"
            expected = canonical_sha256(source) if digest_field == "execution_sha256" else source.get(digest_field)
            if entry[digest_field] != expected:
                raise ReplayError("execution key digest differs")
            seen.add(key)
            selected.append(source)
        return ReplayIndex(
            self.manifest, self.captures, self.systems, self.prefixes, self.entities,
            self.lookups, self.relation_aliases, tuple(selected)
        )

    def for_workload(self, workload: str) -> "ReplayIndex":
        executions = tuple(row for row in self.executions if row.get("workload") == workload)
        declared = [row for row in self.manifest.get("workloads", []) if row.get("name") == workload]
        if not executions or (declared and (
            len(declared) != 1 or declared[0].get("execution_rows") != len(executions)
        )):
            raise ReplayError(f"replay workload {workload!r} is absent or has changed cardinality")
        return ReplayIndex(
            self.manifest, self.captures, self.systems, self.prefixes, self.entities,
            self.lookups, self.relation_aliases, executions
        )

    @staticmethod
    def _record(row: Mapping[str, Any]) -> tuple[str, list[str]]:
        evidence = row["evidence_metadata"]
        parts = [f"CLAIM: {row['claim']}"]
        if row.get("entity_context"):
            parts.append(str(row["entity_context"]))
        if row.get("abbreviation_lines"):
            parts.append("In-text abbreviations:\n" + "\n".join(row["abbreviation_lines"]))
        if row.get("provenance"):
            parts.append(str(row["provenance"]))
        parts.append(f'EVIDENCE: "{evidence["text"]}"')
        return "\n".join(parts), [str(ref) for ref in row.get("lookup_refs", [])]

    def main_request(self, row: Mapping[str, Any], *, relation_note: str = "") -> tuple[str, list[dict[str, str]]]:
        try:
            system = self.systems[str(row["main_system_ref"])]
            prefix = self.prefixes[str(row["main_message_prefix_ref"])]
            user, refs = self._record(row)
            lookups = [self.lookups[ref] for ref in refs]
        except (KeyError, TypeError) as exc:
            raise ReplayError("main prompt references an absent component") from exc
        base_user = user
        if relation_note:
            user += "\n\n" + relation_note
        if lookups:
            user += "\n\nEntity database lookups:\n" + "\n".join(lookups)
        messages = [*(dict(message) for message in prefix), {"role": "user", "content": user}]
        if relation_note:
            insertion = row.get("relation_note_insertion")
            if not isinstance(insertion, Mapping) or (
                insertion.get("message_index") != len(prefix)
                or insertion.get("role") != "user"
                or insertion.get("utf8_byte_offset") != len(base_user.encode("utf-8"))
                or insertion.get("prefix_if_nonempty") != "\n\n"
                or insertion.get("empty_note_inserts_prefix") is not False
            ):
                raise ReplayError("relation-note insertion coordinates differ")
        elif prompt_sha256(system, messages) != row.get("main_prompt_base_sha256"):
            raise ReplayError("hydrated main prompt digest differs")
        return system, messages

    def relation_request(self, row: Mapping[str, Any]) -> tuple[str, list[dict[str, str]]]:
        try:
            system = self.systems[str(row["relation_system_ref"])]
            refs = row["relation_alias_refs"]
            subject = self.relation_aliases[str(refs["subject"])]
            object_ = self.relation_aliases[str(refs["object"])]
            evidence = str(row["evidence_metadata"]["text"])
        except (KeyError, TypeError) as exc:
            raise ReplayError("relation prompt references an absent component") from exc

        def named(name: str, grounding: Mapping[str, Any] | None) -> str:
            aliases = [] if not grounding else [
                alias for alias in grounding["aliases"] if alias.lower() != name.lower()
            ][:6]
            return f"{name} (also known as: {', '.join(aliases)})" if aliases else name

        subject_name, object_name = str(row["subject_name"]), str(row["object_name"])
        user = (
            f"Entities: {named(subject_name, subject)}, {named(object_name, object_)}\n"
            f'Sentence: "{evidence}"\nWhat relationship does the sentence assert between '
            f"{subject_name} and {object_name}?"
        )
        messages = [{"role": "user", "content": user}]
        if prompt_sha256(system, messages) != row.get("relation_prompt_sha256"):
            raise ReplayError("hydrated relation prompt digest differs")
        return system, messages

    def deterministic_result(self, row: Mapping[str, Any]) -> dict[str, Any]:
        route = str(row["route"])
        if route == "no_text":
            return _result(0.95, "correct", "high", "no_text", "skipped", False,
                           "No evidence sentence — accepted by default (database-sourced).", 0, [])
        evidence = str(row["evidence_metadata"]["text"])
        reason = status = None
        pseudogene = False
        for side in ("subject", "object"):
            ref = row["entity_refs"].get(side)
            entity = self.entities.get(str(ref)) if ref else None
            if not entity:
                continue
            mismatch = entity.get("verification_status") == "MISMATCH" and not _entity_in_text(entity, evidence)
            ambiguous_gene = entity.get("verification_status") == "AMBIGUOUS" and entity.get("is_pseudogene") is True and not any(
                token in evidence.lower() for token in ("pseudogene", "lncrna", "lnc-rna", "non-coding rna", "noncoding rna")
            )
            if mismatch or ambiguous_gene:
                pseudogene = bool(entity.get("is_pseudogene"))
                prefix = "Pseudogene mapping" if pseudogene else "Grounding mismatch"
                reason = f"{prefix}: {entity.get('verification_note')}"
                status = str(entity.get("verification_status") or "MISMATCH")
                break
        expected = "deterministic_pseudogene" if pseudogene else "deterministic_mismatch"
        if reason is None or route != expected:
            raise ReplayError("deterministic route no longer reproduces its rejection")
        return _result(0.05, "incorrect", "high", route, status, False, reason, 0, [])


def _entity_in_text(entity: Mapping[str, Any], text: str) -> bool:
    lowered, collapsed = text.lower(), text.lower().replace("-", "").replace(" ", "")
    def contains(value: str) -> bool:
        value = value.lower()
        if len(value) <= 4:
            return bool(re.search(r"(?<![a-z])" + re.escape(value) + r"(?![a-z])", lowered))
        return value in lowered or value.replace("-", "").replace(" ", "") in collapsed
    if contains(str(entity["name"])) or any(len(alias) >= 3 and contains(alias) for alias in entity["all_names"]):
        return True
    raw = entity.get("raw_text")
    if not raw:
        return False
    raw_words = set(str(raw).lower().replace("-", " ").split())
    ignored = {"protein", "factor", "the", "of", "and", "a"}
    return any(len(alias) > 15 and len(raw_words & set(alias.lower().replace("-", " ").split()) - ignored) >= 2
               for alias in entity["all_names"])


def _last_json_object(text: str) -> Mapping[str, Any] | None:
    text = text.strip()
    cut = text.lower().rfind("</think>")
    if cut >= 0:
        text = text[cut + len("</think>"):].strip()
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, Mapping):
        return parsed
    depth = 0
    start = -1
    in_string = escaped = False
    for index, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character == "{":
            start = index if depth == 0 else start
            depth += 1
        elif character == "}" and depth:
            depth -= 1
            if depth == 0:
                try:
                    candidate = json.loads(text[start:index + 1])
                    parsed = candidate if isinstance(candidate, Mapping) else parsed
                except json.JSONDecodeError:
                    pass
    return parsed


def _relation_note(text: str, subject: str, object_: str) -> str:
    parsed = _last_json_object(text)
    if not isinstance(parsed, Mapping) or not isinstance(parsed.get("nature"), str):
        return ""
    nature = re.sub(r"[^a-z]", "", parsed["nature"].lower())
    if not nature or nature == "physicalbinding":
        return ""
    labels = {
        "fusionconstruct": "a gene FUSION / chimeric construct (one molecule)",
        "signalingcascade": "a signaling/regulatory cascade (functional, not physical binding)",
        "cobindingthird": "co-binding to a shared THIRD entity (not each other)",
        "topicoraim": "only a title/topic phrase or an aim/methods clause (not an asserted result)",
        "other": "not a direct physical interaction",
    }
    span = str(parsed.get("span", "") or "")[:160]
    detail = f' — "{span}"' if span else ""
    return (
        f"Relation nature (resolved): the evidence asserts {labels.get(nature, 'not direct physical binding')}{detail}. "
        f"A [Complex] claim requires a stated DIRECT PHYSICAL BIND between {subject} and {object_} — "
        "that is a grounding MISMATCH here, so the [Complex] extraction is unsupported."
    )


def _result(score: float, verdict: str | None, confidence: str | None, tier: str,
            grounding: str | None, provenance: bool, raw: str, tokens: int | None,
            calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"score": score, "verdict": verdict, "confidence": confidence, "tier": tier,
            "grounding_status": grounding, "provenance_triggered": provenance,
            "raw_text": raw, "tokens": tokens, "call_log": [dict(row) for row in calls]}


def score_execution(index: ReplayIndex, row: Mapping[str, Any], client: ClientLike,
                    *, main_max_tokens: int) -> dict[str, Any]:
    route = str(row["route"])
    client.pop_call_log()
    if route not in CALLABLE_ROUTES:
        return index.deterministic_result(row)
    note = ""
    if row.get("relation_prompt_sha256"):
        system, messages = index.relation_request(row)
        try:
            response = client.call(system=system, messages=messages, max_tokens=3000,
                                   temperature=0.1, response_format={"type": "json_object"},
                                   reasoning_effort="none", kind="relation_nature")
            note = _relation_note(response.content or response.raw_text or "",
                                  str(row["subject_name"]), str(row["object_name"]))
        except Exception:
            note = ""
    system, messages = index.main_request(row, relation_note=note)
    kind = "monolithic_tool_context" if route == "tool" else "monolithic"
    response = client.call(system=system, messages=messages, max_tokens=main_max_tokens,
                           temperature=0.1, kind=kind)
    verdict, confidence = parse_response(response)
    trace = getattr(response, "reasoning_trace", None)
    if isinstance(trace, dict):
        parsed = parse_structured(response.content)
        if parsed["verdict"] is None:
            parsed = parse_structured(response.raw_text)
        trace["committed_justification"] = {
            "support": parsed.get("support"), "objection": parsed.get("objection"), "source": "answer_json"
        }
    return _result(verdict_score(verdict, confidence), verdict, confidence,
                   "llm_tool_use" if route == "tool" else "llm_comprehension",
                   "flagged" if route == "tool" else "all_match", bool(row.get("provenance")),
                   response.raw_text, response.tokens, client.pop_call_log())


def source_key(source: Mapping[str, Any]) -> tuple[int, int]:
    return int(source["stmt_i"]), int(source["evidence_i"])


def execution_identity(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "eligible_position": int(source["workload_metadata"]["eligible_position"]),
        "paper_statement_hash": str(source["paper_statement_hash"]),
        "source_hash": str(source["source_hash"]),
        "evidence_json_sha256": str(source["evidence_json_sha256"]),
    }


def expected_execution_id(source: Mapping[str, Any], *, action: Action, model: str) -> str:
    return canonical_sha256({"model": model, "workload_mode": action.workload,
                             **execution_identity(source)})


def _identity_fields(source: Mapping[str, Any], *, action: Action) -> dict[str, Any]:
    evidence = source["evidence_metadata"]
    return {
        "action_id": action.id, "run_id": action.run_id,
        "stmt_i": int(source["stmt_i"]), "evidence_i": int(source["evidence_i"]),
        "source_hash": str(source["source_hash"]),
        "paper_statement_hash": str(source["paper_statement_hash"]),
        "evidence_json_sha256": str(source["evidence_json_sha256"]),
        "statement_type": str(source["statement_type"]),
        "subject": str(source["subject_name"]), "object": str(source["object_name"]),
        "source_api": str(evidence["source_api"]), "pmid": evidence.get("pmid"),
        "text_len": len(str(evidence["text"])),
    }


def result_row(source: Mapping[str, Any], *, action: Action, result: Mapping[str, Any],
               attempt: Mapping[str, Any], latency_s: float) -> dict[str, Any]:
    return {**_identity_fields(source, action=action), **dict(attempt), "row_status": "scored",
            "verdict": result.get("verdict"), "score": result.get("score"),
            "confidence": result.get("confidence"), "tier": result.get("tier"),
            "grounding_status": result.get("grounding_status"),
            "provenance_triggered": result.get("provenance_triggered"),
            "latency_s": round(latency_s, 3), "tokens": result.get("tokens"),
            "call_log": list(result.get("call_log") or []), "error": None,
            "raw_text": str(result.get("raw_text") or "")}


def error_row(source: Mapping[str, Any], *, action: Action,
              calls: Sequence[Mapping[str, Any]], attempt: Mapping[str, Any],
              latency_s: float, error: BaseException | str) -> dict[str, Any]:
    error_type = type(error).__name__ if isinstance(error, BaseException) else str(error)
    message = str(error) if isinstance(error, BaseException) else str(error)
    return {**_identity_fields(source, action=action), **dict(attempt), "row_status": "error",
            "verdict": None, "score": None, "confidence": None, "tier": None,
            "grounding_status": None, "provenance_triggered": None,
            "latency_s": round(latency_s, 3), "tokens": None,
            "call_log": [dict(row) for row in calls],
            "error": {"type": error_type, "message_sha256": _sha_text(message)}, "raw_text": ""}


_ROW_FIELDS = frozenset({
    "action_id", "run_id", "stmt_i", "evidence_i", "source_hash", "paper_statement_hash",
    "evidence_json_sha256", "statement_type", "subject", "object", "source_api", "pmid",
    "text_len", "execution_id", "attempt_id", "attempt_ordinal", "attempt_status",
    "row_status", "verdict", "score", "confidence", "tier", "grounding_status",
    "provenance_triggered", "latency_s", "tokens", "call_log", "error", "raw_text",
})


def validate_row(row: Mapping[str, Any], *, source: Mapping[str, Any], action: Action,
                 model: str, provider_model_id: str) -> None:
    if not isinstance(row, Mapping) or set(row) != _ROW_FIELDS:
        raise ReplayError("raw result row fields differ")
    for field, expected in _identity_fields(source, action=action).items():
        if row.get(field) != expected:
            raise ReplayError(f"raw result identity differs at {field}")
    if row.get("execution_id") != expected_execution_id(source, action=action, model=model):
        raise ReplayError("raw result execution id differs")
    ordinal, attempt_id = row.get("attempt_ordinal"), row.get("attempt_id")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or not 1 <= ordinal <= action.max_attempts:
        raise ReplayError("raw result attempt ordinal is invalid")
    if not isinstance(attempt_id, str) or HEX32.fullmatch(attempt_id) is None:
        raise ReplayError("raw result attempt id is invalid")
    calls = row.get("call_log")
    if not isinstance(calls, list):
        raise ReplayError("raw result call_log is not an array")
    latency, tokens, raw_text = row.get("latency_s"), row.get("tokens"), row.get("raw_text")
    if (
        isinstance(latency, bool) or not isinstance(latency, (int, float))
        or not math.isfinite(float(latency)) or latency < 0
        or (tokens is not None and (
            isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0
        ))
        or not isinstance(raw_text, str)
    ):
        raise ReplayError("raw result latency/token/text fields are malformed")
    expected_topology = list(source["call_topology"])
    for index, call in enumerate(calls, start=1):
        if not isinstance(call, Mapping) or (
            call.get("execution_id") != row["execution_id"]
            or call.get("attempt_id") != attempt_id
            or call.get("attempt_ordinal") != ordinal
            or call.get("call_ordinal") != index
            or call.get("model_id") != provider_model_id
            or index > len(expected_topology)
            or call.get("kind") != expected_topology[index - 1]
            or not isinstance(call.get("call_id"), str)
            or HEX32.fullmatch(call["call_id"]) is None
            or not isinstance(call.get("provider_request_sha256"), str)
            or HEX64.fullmatch(call["provider_request_sha256"]) is None
        ):
            raise ReplayError("raw result provider call identity/topology differs")
    if row.get("row_status") == "scored":
        if row.get("attempt_status") != "completed" or row.get("error") is not None:
            raise ReplayError("scored row has contradictory attempt status")
        if source["route"] in CALLABLE_ROUTES and len(calls) != len(expected_topology):
            raise ReplayError("scored row lacks its complete provider topology")
        if source["route"] not in CALLABLE_ROUTES and calls:
            raise ReplayError("deterministic scored row contains provider calls")
        if row.get("verdict") not in {None, "correct", "incorrect"} or row.get("confidence") not in {None, "high", "medium", "low"}:
            raise ReplayError("scored verdict/confidence is invalid")
        if row.get("score") != verdict_score(row.get("verdict"), row.get("confidence")):
            raise ReplayError("scored row score differs from verdict/confidence")
        route = str(source["route"])
        expected_tier = (
            "llm_tool_use" if route == "tool" else
            "llm_comprehension" if route == "plain" else route
        )
        if (
            row.get("tier") != expected_tier
            or not isinstance(row.get("provenance_triggered"), bool)
            or (route in CALLABLE_ROUTES and tokens is None)
        ):
            raise ReplayError("scored row tier/provenance/tokens differ from its route")
    elif row.get("row_status") == "error":
        error = row.get("error")
        if row.get("attempt_status") not in {"error", "scheduling_stopped", "reconciled_after_interruption"} or not isinstance(error, Mapping) or set(error) != {"type", "message_sha256"} or not isinstance(error["type"], str) or HEX64.fullmatch(str(error["message_sha256"])) is None:
            raise ReplayError("error row failure commitment is malformed")
        if any(row.get(field) is not None for field in (
            "verdict", "score", "confidence", "tier", "grounding_status",
            "provenance_triggered", "tokens",
        )) or raw_text:
            raise ReplayError("error row carries scored-result fields")
    else:
        raise ReplayError("raw result row_status is invalid")


@dataclass(frozen=True)
class ResumeState:
    status: str
    rows: tuple[Mapping[str, Any], ...]
    latest: Mapping[tuple[int, int], Mapping[str, Any]]
    done: frozenset[tuple[int, int]]
    attempts: Mapping[tuple[int, int], int]
    verdicts: Mapping[str, int]


def _terminal_row(row: Mapping[str, Any]) -> bool:
    return (
        row.get("row_status") == "scored"
        and row.get("verdict") in {"correct", "incorrect"}
    )


def load_resume(path: Path, *, index: ReplayIndex, action: Action, model: str,
                provider_model_id: str, stream: Any | None = None) -> ResumeState:
    """Load append-only attempts without imposing a cross-source row order.

    Attempts must be contiguous for each source, but concurrent sources may be
    appended in any completion order.  Historical scored rows with no valid
    verdict remain immutable attempt evidence; they are nonterminal and may be
    followed by the next bounded attempt.
    """
    if stream is None and not path.exists():
        return ResumeState("pending", (), {}, frozenset(), {}, {})
    if stream is None:
        raw = stable_read(path, context="raw output").payload
    else:
        os.lseek(stream.fileno(), 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(stream.fileno(), 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        os.lseek(stream.fileno(), 0, os.SEEK_END)
    if raw and not raw.endswith(b"\n"):
        raise ReplayError("raw output ends in a partial JSONL row")
    sources = {source_key(row): row for row in index.executions}
    rows: list[Mapping[str, Any]] = []
    latest: dict[tuple[int, int], Mapping[str, Any]] = {}
    attempts: Counter[tuple[int, int]] = Counter()
    call_ids: set[str] = set()
    for number, line in enumerate(raw.splitlines(keepends=True), start=1):
        value = strict_json_loads(line[:-1], context=f"raw output row {number}")
        if not isinstance(value, Mapping) or canonical_json_line(value) != line:
            raise ReplayError(f"raw output row {number} is not canonical JSON")
        key = (value.get("stmt_i"), value.get("evidence_i"))
        source = sources.get(key)
        if source is None:
            raise ReplayError(f"raw output row {number} is outside the action")
        attempts[key] += 1
        if value.get("attempt_ordinal") != attempts[key] or attempts[key] > action.max_attempts:
            raise ReplayError("raw output attempts are not a bounded contiguous prefix")
        if key in latest and _terminal_row(latest[key]):
            raise ReplayError("raw output appends an attempt after a terminal scored row")
        validate_row(value, source=source, action=action, model=model,
                     provider_model_id=provider_model_id)
        ids = {str(call["call_id"]) for call in value["call_log"]}
        if len(ids) != len(value["call_log"]) or ids & call_ids:
            raise ReplayError("raw output repeats a provider call id")
        call_ids.update(ids)
        latest[key] = value
        rows.append(value)
    done = frozenset(key for key, row in latest.items() if _terminal_row(row))
    status = "complete" if len(done) == len(sources) else ("partial" if rows else "pending")
    verdicts = Counter(str(row["verdict"]) for row in latest.values()
                       if _terminal_row(row))
    return ResumeState(status, tuple(rows), latest, done, dict(attempts), dict(verdicts))


@dataclass
class AppendLog:
    path: Path
    stream: Any
    identity: tuple[int, int]

    @classmethod
    def open(cls, path: Path) -> "AppendLog":
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags, 0o600)
        status = os.fstat(fd)
        if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
            os.close(fd)
            raise ReplayError("raw output is not a unique regular file")
        stream = os.fdopen(fd, "a+b", buffering=0)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise ReplayError("raw output is already in use") from exc
        return cls(path, stream, (status.st_dev, status.st_ino))

    def assert_current(self) -> None:
        held = os.fstat(self.stream.fileno())
        named = os.stat(self.path, follow_symlinks=False)
        if (held.st_dev, held.st_ino) != self.identity or (named.st_dev, named.st_ino) != self.identity or held.st_nlink != 1 or named.st_nlink != 1:
            raise ReplayError("raw output left its pinned inode")

    def append(self, row: Mapping[str, Any]) -> None:
        self.assert_current()
        payload = canonical_json_line(row)
        if self.stream.write(payload) != len(payload):
            raise ReplayError("short append to raw output")
        os.fsync(self.stream.fileno())

    def close(self) -> None:
        self.stream.close()
