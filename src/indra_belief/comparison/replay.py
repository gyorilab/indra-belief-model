"""Provider-free prompt replay, raw-row validation, and append-only resume.

This module no longer ASSEMBLES a request. `ReplayIndex` owns the
content-addressed component tables and resolves a row's refs against them;
`indra_belief.prepared_execution` owns the request itself and the digest
contract the frozen substrate commits to. `ReplayError` and `prompt_sha256`
moved there with it and are re-exported here, so every existing importer and
every `except ReplayError` still resolves.

It no longer READS one either. `indra_belief.verdict` owns the parser for this
path and for the live scorer alike. Frozen raw rows are integrity-checked by
their hash-chained spend WAL, so replay does not need to re-derive an obsolete
display score from verdict/confidence. New replay rows carry `score=None`: the
calibrated forced-verdict probe is not representable by the guarded provider
client, and absence must remain explicit.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import stat
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol, Sequence

from indra_belief.prepared_execution import (
    RELATION_CALL_KIND,
    PreparedCall,
    PreparedExecution,
    ReplayError,
    assert_replay_digests,
    prepare_from_replay_row,
    prompt_sha256,
    relation_mismatch_note,
    relation_user_message,
)
from indra_belief.spend_guard import classify_provider_failure
from indra_belief.verdict import NO_TEXT_RESULT, parse_response

from .contracts import (
    Action,
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
_FORBIDDEN_INPUT_KEYS = frozenset({
    "belief", "correct", "curation", "curations", "curator", "curator_note",
    "gold", "incorrect", "label", "labels", "model_output", "model_outputs",
    "model_response", "model_responses", "predicted_label", "prediction",
    "predictions", "tag", "tags", "verdict", "verdicts",
})


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
                # `prepare` resolves every ref (and digest-checks the relation
                # sub-call when the row carries one); this is the main prompt's
                # own commitment.
                assert_replay_digests(self.prepare(row), row)
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

    def prepare(self, row: Mapping[str, Any], *,
                max_tokens: int | None = None) -> PreparedExecution:
        """Resolve this row's component refs into the one request value.

        Ref resolution is all that is left here: the index owns the
        content-addressed `systems` / `prefixes` / `lookups` tables, and
        `prepare_from_replay_row` owns the request. Nothing is digest-checked
        yet — `assert_replay_digests` does that, for both callers.
        """
        contract = self.manifest.get("generation_contract")
        execution = prepare_from_replay_row(
            row, systems=self.systems, prefixes=self.prefixes, lookups=self.lookups,
            max_tokens=max_tokens,
            profile_name=str(contract.get("mono_variant", ""))
            if isinstance(contract, Mapping) else "",
        )
        if not row.get("relation_prompt_sha256"):
            return execution
        system, messages = self.relation_request(row)
        return replace(execution, relation=PreparedCall(
            kind=RELATION_CALL_KIND, system=system, messages=tuple(messages),
            max_tokens=3000, temperature=0.1,
            response_format={"type": "json_object"}, reasoning_effort="none",
        ))

    def relation_request(self, row: Mapping[str, Any]) -> tuple[str, list[dict[str, str]]]:
        try:
            system = self.systems[str(row["relation_system_ref"])]
            refs = row["relation_alias_refs"]
            subject = self.relation_aliases[str(refs["subject"])]
            object_ = self.relation_aliases[str(refs["object"])]
            evidence = str(row["evidence_metadata"]["text"])
        except (KeyError, TypeError) as exc:
            raise ReplayError("relation prompt references an absent component") from exc

        subject_name, object_name = str(row["subject_name"]), str(row["object_name"])
        user = relation_user_message(
            subject_name, object_name, evidence, subject, object_
        )
        messages = [{"role": "user", "content": user}]
        if prompt_sha256(system, messages) != row.get("relation_prompt_sha256"):
            raise ReplayError("hydrated relation prompt digest differs")
        return system, messages

    def deterministic_result(self, row: Mapping[str, Any]) -> dict[str, Any]:
        route = str(row["route"])
        if route == "no_text":
            return {**NO_TEXT_RESULT, "call_log": []}
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
        return _result(None, "incorrect", "high", route, status, False, reason, 0, [])


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
    return relation_mismatch_note(
        nature, parsed.get("span", "") or "", subject, object_
    )


def _result(score: float | None, verdict: str | None, confidence: str | None, tier: str,
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
    execution = index.prepare(row, max_tokens=main_max_tokens)
    note = ""
    if execution.relation is not None:
        try:
            response = client.call(**execution.relation.client_kwargs())
            note = _relation_note(response.content or response.raw_text or "",
                                  str(row["subject_name"]), str(row["object_name"]))
        except Exception:
            note = ""
    assert_replay_digests(execution, row, relation_note=note)
    response = client.call(**execution.calls(note)[-1].client_kwargs())
    parsed = parse_response(response)
    verdict = None if parsed is None else parsed.label
    confidence = None if parsed is None else parsed.confidence
    trace = getattr(response, "reasoning_trace", None)
    if isinstance(trace, dict):
        trace["committed_justification"] = {
            "support": None if parsed is None else parsed.support,
            "objection": None if parsed is None else parsed.objection,
            "source": "answer_json",
        }
    return _result(None, verdict, confidence,
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
    # `provider_http_status` is recorded because `error.type` alone cannot
    # separate the failures the settled boundary must treat differently: both an
    # auth 401 and a malformed-request 400 surface as `BedrockChatTransportError`,
    # and `message_sha256` destroys the text the live classifier reads the status
    # out of. Without it the resume path has no way to know that a restart with a
    # fresh token would have scored the source. Live decides, the row records; the
    # resume path reads rather than re-derives.
    status = classify_provider_failure(error)[1] if isinstance(error, BaseException) else None
    return {**_identity_fields(source, action=action), **dict(attempt), "row_status": "error",
            "verdict": None, "score": None, "confidence": None, "tier": None,
            "grounding_status": None, "provenance_triggered": None,
            "latency_s": round(latency_s, 3), "tokens": None,
            "call_log": [dict(row) for row in calls],
            "error": {"type": error_type, "message_sha256": _sha_text(message),
                      "provider_http_status": status}, "raw_text": ""}


# Rows written before `provider_http_status` existed carry two keys and stay
# valid forever — they are append-only durable evidence and may not be rewritten.
# Measured at the commit that added the third key: all 1,653 error rows across the
# 15 shipped attempt logs carry the two-key shape.
_ERROR_COMMITMENT_SHAPES = (
    {"type", "message_sha256"},
    {"type", "message_sha256", "provider_http_status"},
)


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
        # Historical rows may contain the retired display score. Their exact
        # bytes are already committed by the hash-chained WAL; only keep a
        # generic schema/domain check here. Newly generated replay rows use None
        # because the calibrated probe is unavailable through this guarded path.
        score = row.get("score")
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise ReplayError("scored row score is neither null nor a probability")
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
        if row.get("attempt_status") not in {"error", "scheduling_stopped", "reconciled_after_interruption"} or not isinstance(error, Mapping) or set(error) not in _ERROR_COMMITMENT_SHAPES or not isinstance(error["type"], str) or HEX64.fullmatch(str(error["message_sha256"])) is None:
            raise ReplayError("error row failure commitment is malformed")
        if "provider_http_status" in error and not (
            error["provider_http_status"] is None
            or isinstance(error["provider_http_status"], int)
        ):
            raise ReplayError("error row provider_http_status is not an integer or null")
        if any(row.get(field) is not None for field in (
            "verdict", "score", "confidence", "tier", "grounding_status",
            "provenance_triggered", "tokens",
        )) or raw_text:
            raise ReplayError("error row carries scored-result fields")
    else:
        raise ReplayError("raw result row_status is invalid")


# Per-source cap on unparseable model output, counted cumulatively across
# restarts from the persisted rows (see ResumeState.invalid_outputs). It exists
# to stop a model that CANNOT do the task from burning the whole transport-retry
# budget, and is deliberately separate from, and much tighter than,
# action.max_attempts.
#
# Raised 2 -> 5 on 2026-07-31 against measured evidence from the verdict-only
# run. That prompt asks for two closed-enum fields, and every one of the four
# models occasionally fills them in the wrong order — e.g. gemma-4-e2b returned
# `{"verdict": "medium", "confidence": "medium"}`. The rate is 0.057%-0.097%
# across all four arms, and it is sporadic rather than systematic: re-issuing
# the SAME request for the execution that wedged returned a valid verdict on 2
# of 3 tries. At a per-request failure probability of about a third for a hard
# input, a cap of 2 retires a source 11% of the time, and back when the first
# retired source halted the whole arm that is exactly what happened; gemma_26b
# finished carrying 19 such errors purely by luck. Five leaves ~0.4%.
#
# This bound is PER SOURCE, and per source only.  Since quarantine landed it no
# longer stops a systematically broken model: five attempts retire the first
# source and the scheduler moves to the next one, so the systematic case is
# bounded by `runner.QUARANTINE_FLOOR`/`QUARANTINE_MAX_FRACTION` — the aggregate
# circuit breaker — and not by anything here.  It lives here rather than in the
# runner because it is a property of the durable rows: `_settled_reason` below is
# the only reader that decides anything with it, and the runner consumes that
# decision.
INVALID_MODEL_OUTPUT_LIMIT = 5


def resolved_status(*, done: int, settled: int, total: int, any_rows: bool) -> str:
    """The four action states, from the two disjoint counts of resolved sources.

    ONE settled source is enough to make the action "settled", even with tens of
    thousands still schedulable, and that is the whole point.  This corpus is
    all-or-nothing: `llm._load_pairs` and `_validate_raw` require a bundle to
    cover the exact 1,689-statement / 33,361-execution universe, so an action
    holding a single unscored source can never be published no matter how much
    more of it is scored.  Continuing to dispatch it spends real money on an
    artifact that cannot exist.  "Settled" says: this action is finished, it did
    not finish cleanly, and the cause has to be fixed rather than out-run.

    `settled` is the state S2 had no name for.  Before it existed such an action
    reported "partial" forever, so `contracts.ready_actions` re-offered it, every
    dependent deadlocked, and each futile invocation demanded a live bearer token
    to build zero clients and issue zero calls.

    "settled" and "complete" are deliberately DISTINCT and must never be merged.
    Only "complete" means "every source carries a verdict", and that is what the
    bundle gate, the report and the paper all rest on; anything asking whether an
    action finished CLEANLY must keep testing for that one string.
    """
    if done == total:
        return "complete"
    if settled:
        return "settled"
    return "partial" if any_rows else "pending"


@dataclass(frozen=True)
class ResumeState:
    status: str
    rows: tuple[Mapping[str, Any], ...]
    latest: Mapping[tuple[int, int], Mapping[str, Any]]
    done: frozenset[tuple[int, int]]
    attempts: Mapping[tuple[int, int], int]
    invalid_outputs: Mapping[tuple[int, int], int]
    settled: Mapping[tuple[int, int], str]
    verdicts: Mapping[str, int]
    # The `error.type` of the row that settled each key, for the keys that have
    # one. Additive and parallel to `settled` rather than folded into its value,
    # so every existing reader of `settled` — the scheduler's last gate, the
    # summary, the tests — keeps reading a plain reason string.
    settled_error_types: Mapping[tuple[int, int], str] = field(default_factory=dict)


def _terminal_row(row: Mapping[str, Any]) -> bool:
    return (
        row.get("row_status") == "scored"
        and row.get("verdict") in {"correct", "incorrect"}
    )


def row_retry_class(row: Mapping[str, Any]) -> str | None:
    if row.get("row_status") == "scored":
        return (
            None
            if row.get("verdict") in {"correct", "incorrect"}
            else "invalid_model_output"
        )
    if row.get("row_status") != "error":
        return None
    calls = row.get("call_log")
    if (
        isinstance(calls, list)
        and calls
        and calls[-1].get("provider_failure_class") == "transport_or_server"
    ):
        return "transport_or_server"
    error = row.get("error")
    if not isinstance(error, Mapping):
        return None
    if error.get("type") == "InvalidModelOutput":
        return "invalid_model_output"
    if error.get("provider_http_status") in _CREDENTIAL_STATUSES:
        return "credential"
    # Resume classification is name-based while live classification is
    # isinstance-based: ActionDeadlineExceeded (a TimeoutError), WAL
    # interruption closures, and reservation-accounting breaches record
    # operational events, not provider verdicts, and must stay retryable
    # across restarts.
    if error.get("type") in _RETRYABLE_ERROR_TYPES:
        return "transport_or_server"
    return None


# 401 and 403. The live classifier calls both "other" and does not retry them
# WITHIN an attempt loop, which is right — the same token will be rejected again
# in the same process. Across a RESTART it is wrong: `_run_prepared` re-reads the
# token, so the durable row does not answer "another attempt is not permitted",
# only "the token at that moment was rejected". Settling on it is what turns a
# credential blip near the end of a run into a permanent hole, and any hole is
# terminal. Being wrong here costs one free, fast call and then halts the action
# exactly as it does today; being wrong the other way costs the arm.
_CREDENTIAL_STATUSES = frozenset({401, 403})


def _provider_error_names() -> frozenset[str]:
    """Transport exception NAMES the live classifier retries by `isinstance`.

    `classify_provider_failure` decides on the live path with
    `isinstance(error, (TimeoutError, ConnectionError))`; this path holds only
    `error.type`, a string, so it has to decide by name. Typing the names out is
    what let the two drift — `BedrockChatConnectionError` IS a `ConnectionError`
    and the live path has always retried it, while the hand-written set below
    never named it. Deriving the names from the classes closes that: a new
    transport exception subclassing `ConnectionError` joins both classifiers in
    the same commit that defines it.

    Both transport modules are stdlib-only (~40 ms to import) so this costs
    nothing the runner does not already pay.
    """
    from indra_belief import bedrock_chat_transport, bedrock_responses_transport

    names = set()
    for module in (bedrock_chat_transport, bedrock_responses_transport):
        for value in vars(module).values():
            if (
                isinstance(value, type)
                and issubclass(value, BaseException)
                and issubclass(value, (TimeoutError, ConnectionError))
            ):
                names.add(value.__name__)
    return frozenset(names)


# The three that are NOT provider transports and must stay hand-named: the first
# two are defined in `comparison.runner`, which imports THIS module, and the
# third in `spend_guard` as a `RuntimeError`. All three record operational
# events rather than provider verdicts.
_RETRYABLE_ERROR_TYPES = _provider_error_names() | {
    "TimeoutError",
    "ConnectionError",
    "ActionDeadlineExceeded",
    "InterruptedAfterDurableProviderEvidence",
    "SpendReservationBreach",
}


def _scan_resume(path: Path, *, index: ReplayIndex, action: Action, model: str,
                 provider_model_id: str, stream: Any | None = None,
                 ) -> Iterator[tuple[tuple[Any, Any], Mapping[str, Any]]]:
    """Parse and validate the append-only attempt rows, yielding (key, row).

    The single owner of the resume parse checks.  `load_resume` and
    `resume_status` differ only in what they RETAIN from this stream, so neither
    can drift into a second, laxer validation chain: a row is accepted by both
    or by neither, and a corrupt file raises the same `ReplayError` for both.
    Consume it to exhaustion — a short-circuiting fold would skip later checks.
    """
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
    attempts: Counter[tuple[int, int]] = Counter()
    terminal: dict[tuple[int, int], bool] = {}
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
        if terminal.get(key):
            raise ReplayError("raw output appends an attempt after a terminal scored row")
        validate_row(value, source=source, action=action, model=model,
                     provider_model_id=provider_model_id)
        ids = {str(call["call_id"]) for call in value["call_log"]}
        if len(ids) != len(value["call_log"]) or ids & call_ids:
            raise ReplayError("raw output repeats a provider call id")
        call_ids.update(ids)
        yield key, value
        terminal[key] = _terminal_row(value)


def _settled_reason(latest: Mapping[str, Any], *, attempts: int, invalid_outputs: int,
                    max_attempts: int) -> str | None:
    """Why a nonterminal source can never be scored again, or None if it can.

    "Settled" is a strictly weaker predicate than `_terminal_row`, and the two
    must stay separate.  `_terminal_row` decides what may be APPENDED after
    (`_scan_resume` rejects an attempt following a terminal row), so widening it
    to cover exhausted sources would reject every legitimate retry.  Settled
    decides only what may be SCHEDULED: the durable rows already answer "another
    attempt is not permitted", so re-dispatching the source can only re-pay for
    the identical refusal.

    It is derived, never persisted.  Every input is a fold over the same
    append-only rows plus `action.max_attempts`, so a restart recomputes the
    identical set and no crash can silently un-settle — or silently settle — a
    source.  Order matters only for which reason is REPORTED; the boundary
    itself is the disjunction.
    """
    return _settled_reason_from_class(
        row_disposition(latest), attempts=attempts,
        invalid_outputs=invalid_outputs, max_attempts=max_attempts,
    )


@dataclass(frozen=True)
class RowDisposition:
    """What one durable row says about its own failure, for the settled boundary.

    `retry_class` is `row_retry_class`'s answer, unchanged. `error_type` rides
    beside it because the boundary used to throw it away: an auth failure, a
    config error, a parser-profile mismatch and an unclassified exception all
    collapsed into the single reason `nonretryable_failure_on_resume`, so the
    quarantine record could not say WHICH of them retired a source. The kind
    stays in `runner._QUARANTINE_KINDS` — that allowlist is exact-match and its
    default is halt, so it must not learn to match a prefix — and the type is
    reported alongside it instead.
    """

    retry_class: str | None
    error_type: str | None


def row_disposition(row: Mapping[str, Any]) -> RowDisposition:
    error = row.get("error")
    error_type = error.get("type") if isinstance(error, Mapping) else None
    return RowDisposition(
        row_retry_class(row),
        str(error_type) if isinstance(error_type, str) else None,
    )


def _settled_reason_from_class(disposition: RowDisposition, *, attempts: int,
                               invalid_outputs: int, max_attempts: int) -> str | None:
    """The settled boundary itself, over the fold rather than over the row.

    `load_resume` holds the latest row and `resume_status` deliberately does not,
    so the shared decision is expressed on the only thing both can produce: the
    disposition of that row plus the three counts.  One boundary, two callers.
    """
    if disposition.retry_class == "credential":
        # NOT settled, and this is the one class where that is a change. See
        # `_CREDENTIAL_STATUSES`: the token is re-read at restart, so the row
        # does not say another attempt is impossible. It is still bounded by
        # `max_attempts` below, like everything else.
        return (
            "attempts_exhausted" if attempts >= max_attempts else None
        )
    if disposition.retry_class is None:
        return "nonretryable_failure_on_resume"
    if invalid_outputs >= INVALID_MODEL_OUTPUT_LIMIT:
        return "invalid_model_output_limit"
    if attempts >= max_attempts:
        return "attempts_exhausted"
    return None


def load_resume(path: Path, *, index: ReplayIndex, action: Action, model: str,
                provider_model_id: str, stream: Any | None = None) -> ResumeState:
    """Load append-only attempts without imposing a cross-source row order.

    Attempts must be contiguous for each source, but concurrent sources may be
    appended in any completion order.  Historical scored rows with no valid
    verdict remain immutable attempt evidence; they are nonterminal and may be
    followed by the next bounded attempt.
    """
    if stream is None and not path.exists():
        return ResumeState("pending", (), {}, frozenset(), {}, {}, {}, {})
    rows: list[Mapping[str, Any]] = []
    latest: dict[tuple[int, int], Mapping[str, Any]] = {}
    attempts: Counter[tuple[int, int]] = Counter()
    invalid: Counter[tuple[int, int]] = Counter()
    for key, value in _scan_resume(path, index=index, action=action, model=model,
                                   provider_model_id=provider_model_id, stream=stream):
        latest[key] = value
        rows.append(value)
        attempts[key] += 1
        if row_retry_class(value) == "invalid_model_output":
            invalid[key] += 1
    total = len({source_key(row) for row in index.executions})
    done = frozenset(key for key, row in latest.items() if _terminal_row(row))
    verdicts = Counter(str(row["verdict"]) for row in latest.values()
                       if _terminal_row(row))
    settled: dict[tuple[int, int], str] = {}
    settled_error_types: dict[tuple[int, int], str] = {}
    for key, row in latest.items():
        if key in done:
            continue
        reason = _settled_reason(row, attempts=attempts[key],
                                 invalid_outputs=invalid[key],
                                 max_attempts=action.max_attempts)
        if reason is not None:
            settled[key] = reason
            error_type = row_disposition(row).error_type
            if error_type is not None:
                settled_error_types[key] = error_type
    status = resolved_status(done=len(done), settled=len(settled), total=total,
                             any_rows=bool(rows))
    return ResumeState(status, tuple(rows), latest, done, dict(attempts),
                       dict(invalid), settled, dict(verdicts), settled_error_types)


@dataclass(frozen=True)
class ResumeStatus:
    status: str
    completed: int
    attempts: int
    settled: int = 0


def resume_status(path: Path, *, index: ReplayIndex, action: Action, model: str,
                  provider_model_id: str) -> ResumeStatus:
    """Answer only what readiness asks, without retaining the parsed rows.

    `prepare_run` and `inspect_plan` run this over EVERY action in the plan but
    read only a status string and three counts, so retaining the whole file per
    action is pure cost.  This runs the identical `_scan_resume` checks and keeps
    four scalars per source key — the same fold `load_resume` performs, minus the
    rows.  Both must reach the identical status for the same bytes, so the
    settled counts are derived here rather than approximated; a status that
    disagreed with `load_resume` would let readiness and the scheduler describe
    different actions.  Callers that need the rows themselves — `_recover` and
    `_reconcile` — must still use `load_resume`.
    """
    if not path.exists():
        return ResumeStatus("pending", 0, 0, 0)
    rows_seen = 0
    terminal: dict[tuple[int, int], bool] = {}
    disposition: dict[tuple[int, int], RowDisposition] = {}
    attempts: Counter[tuple[int, int]] = Counter()
    invalid: Counter[tuple[int, int]] = Counter()
    for key, value in _scan_resume(path, index=index, action=action, model=model,
                                   provider_model_id=provider_model_id):
        rows_seen += 1
        terminal[key] = _terminal_row(value)
        disposition[key] = row_disposition(value)
        attempts[key] += 1
        if disposition[key].retry_class == "invalid_model_output":
            invalid[key] += 1
    total = len({source_key(row) for row in index.executions})
    completed = sum(terminal.values())
    settled = sum(
        1
        for key, is_terminal in terminal.items()
        if not is_terminal
        and _settled_reason_from_class(
            disposition[key], attempts=attempts[key],
            invalid_outputs=invalid[key], max_attempts=action.max_attempts,
        ) is not None
    )
    status = resolved_status(done=completed, settled=settled, total=total,
                             any_rows=bool(rows_seen))
    return ResumeStatus(status, completed, rows_seen, settled)


# One attempt row is a bounded JSON object; the largest in the shipped corpus is
# well under this. The bound's job is to separate "a torn append" from "something
# else wrote here", and it errs generously so a legitimately large row is never
# mistaken for the second.
_TORN_TAIL_LIMIT = 4 * 1024 * 1024


@dataclass(frozen=True)
class TornTail:
    """What a recovery discarded. The audit record, not a log line.

    A truncation that leaves no trace is indistinguishable from a run that never
    had a torn tail, and those are different histories: the first lost a paid
    attempt to a crash. The digest is of the discarded fragment, so the bytes
    can be recognised again if they turn up in an operator's copy.
    """

    bytes_discarded: int
    sha256: str
    size_before: int
    size_after: int

    def describe(self) -> str:
        return (
            f"discarded a {self.bytes_discarded}-byte unterminated trailing record "
            f"(sha256 {self.sha256}); {self.size_before} -> {self.size_after} bytes"
        )


@dataclass
class AppendLog:
    path: Path
    stream: Any
    identity: tuple[int, int]
    # What `open(recover=True)` discarded, or None. Never a silent repair: a
    # caller that truncates a durable log has to be able to say what it removed.
    recovered: "TornTail | None" = None

    @classmethod
    def open(cls, path: Path, *, recover: bool = False) -> "AppendLog":
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
        log = cls(path, stream, (status.st_dev, status.st_ino))
        if recover:
            try:
                log.recovered = log._recover_torn_tail()
            except BaseException:
                stream.close()
                raise
        return log

    def _recover_torn_tail(self) -> "TornTail | None":
        """Discard a partial trailing record, or return None if there is none.

        WHY THIS IS THE ONLY PLACE IT MAY HAPPEN. Truncating a durable
        ledger-backed log is a real mutation of paid evidence, so it may be done
        only by the process that is about to append to it, holding `LOCK_EX`,
        on a descriptor pinned to a verified inode — all three of which are true
        here and nowhere else. `load_resume` and `resume_status` keep raising
        `ReplayError` on a torn tail, unchanged: a READER must never repair, and
        the fifteen published logs are read by tools that hold no lock.

        WHY A TRAILING PARTIAL IS RECOVERABLE AT ALL. The log is append-only and
        every record is one unbuffered write followed by `fsync`, so a torn
        record can only ever be the last one. `SIGKILL` cannot tear it — the
        write is a single syscall — but power loss between the page cache
        landing and the fsync can, and so can the short-append branch of
        `append`, which raises AFTER its bytes are on disk. Both leave the same
        shape: complete records, then a fragment with no newline.

        THREE REFUSALS, because "the tail has no newline" is consistent with
        faults that are NOT a torn append and that truncation would destroy
        evidence of:

          * a fragment larger than `_TORN_TAIL_LIMIT` is not a torn append. One
            record is a bounded JSON row; megabytes of tail means something
            else wrote here.
          * a fragment that parses as complete canonical JSON is a LOST
            NEWLINE, not a lost record — the row is intact and discarding it
            would throw away a paid attempt.
          * a file with no newline anywhere is not a partial tail with complete
            records before it; it is a file whose entire content is unframed.
        """
        self.assert_current()
        size = os.fstat(self.stream.fileno()).st_size
        if size == 0:
            return None
        # Two past the bound, so a fragment of exactly `_TORN_TAIL_LIMIT + 1` is
        # visible TOGETHER WITH the newline that precedes it. Read a smaller
        # window and an over-long fragment pushes that newline outside it, which
        # is indistinguishable from a file with no boundary at all — a wrong
        # diagnosis, and the one this window size exists to prevent.
        read_from = max(0, size - (_TORN_TAIL_LIMIT + 2))
        tail = os.pread(self.stream.fileno(), size - read_from, read_from)
        if tail.endswith(b"\n"):
            return None
        boundary = tail.rfind(b"\n")
        if boundary < 0:
            if read_from == 0:
                raise ReplayError(
                    "raw output has no record boundary before its unterminated tail"
                )
            raise ReplayError(
                f"raw output's unterminated tail exceeds the {_TORN_TAIL_LIMIT}-byte "
                "bound for one torn record"
            )
        fragment = tail[boundary + 1:]
        if len(fragment) > _TORN_TAIL_LIMIT:
            raise ReplayError(
                f"raw output's unterminated tail is {len(fragment)} bytes, over the "
                f"{_TORN_TAIL_LIMIT}-byte bound for one torn record"
            )
        try:
            value = json.loads(fragment)
        except ValueError:
            value = None
        if isinstance(value, Mapping) and canonical_json_line(value) == fragment + b"\n":
            raise ReplayError(
                "raw output's tail is a complete canonical row missing only its "
                "newline; that is a lost terminator, not a lost record, and "
                "discarding it would drop a paid attempt"
            )
        keep = read_from + boundary + 1
        os.ftruncate(self.stream.fileno(), keep)
        os.fsync(self.stream.fileno())
        return TornTail(
            bytes_discarded=len(fragment),
            sha256=hashlib.sha256(fragment).hexdigest(),
            size_before=size,
            size_after=keep,
        )

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
