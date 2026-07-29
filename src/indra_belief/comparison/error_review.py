"""Canonical, offline, human-only review of threshold errors.

Review material is derived only from a hash-bound LLM model bundle and the
exact evidence-execution identities used by its selected panel.  Reviewer
artifacts contain opaque keyed identifiers and scrubbed claim/evidence only;
the administrator manifest retains the identity mapping and full provenance.
No function in this module calls a model or fabricates a human decision.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Iterable, Mapping, NoReturn, Sequence

from .contracts import ContractError, canonical_json_bytes, stable_read, strict_json_loads
from .llm import READER_SOURCES


PACKET_KIND = "indra_belief_error_review_packet"
ADMIN_KIND = "indra_belief_error_review_admin_manifest"
CODEBOOK_KIND = "indra_belief_error_review_codebook"
LEDGER_KIND = "indra_belief_error_review_ledger"
WORKBOOK_KIND = "indra_belief_error_review_workbook"
RESOLVER_WORKLOAD_KIND = "indra_belief_error_review_resolver_workload"
REPORT_KIND = "indra_belief_error_review_report"
CANONICAL_PROTOCOL_SHA256 = (
    "910b660d626202668f72c941f277ebee95fb8794e83208082995c58a4fe1987a"
)

HUMAN_ATTESTATION = (
    "I attest that I personally reviewed every assigned case without "
    "model-generated adjudication and that this ledger accurately records my decisions."
)
FREEZE_ATTESTATION = (
    "I attest that humans completed the bound pilot and that this frozen codebook "
    "records the resulting classification rubric and dimension taxonomy."
)
CLASSIFICATIONS = ("supports_claim", "rejects_claim", "indeterminate")
CLASSIFICATION_DEFINITIONS = {
    "supports_claim": (
        "The exact displayed material establishes the assembled claim under the frozen rubric."
    ),
    "rejects_claim": (
        "The exact displayed material contradicts the assembled claim or is insufficient "
        "to establish it under the frozen rubric."
    ),
    "indeterminate": (
        "The exact displayed material is materially insufficient, conflicting, or ambiguous, "
        "so it does not warrant a binary supports-claim or rejects-claim classification."
    ),
}
CLASSIFICATION_STEPS = [
    "Use only the exact displayed claim and evidence; do not use a model or external lookup.",
    "Classify whether the displayed material supports or rejects the assembled claim without guessing which system produced it.",
    "Choose indeterminate when ambiguity, conflict, or missing information makes either binary classification unsafe.",
    "Assign one or more causal dimensions and use the optional comment for taxonomy refinements.",
]
PILOT_CONTRACT = {
    "selection": "keyed deterministic, direction-balanced sample of threshold errors",
    "required_independent_reviewers": 2,
    "freeze_requires": (
        "two complete human pilot ledgers and an explicit human freeze attestation"
    ),
}
DIMENSION_ID = re.compile(r"[a-z][a-z0-9_]{1,63}\Z")
PSEUDONYM = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,63}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
UUID = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z"
)
UUID_TOKEN = re.compile(
    r"(?<![0-9a-fA-F])[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}(?![0-9a-fA-F])"
)
HEX64_TOKEN = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])")
OPAQUE_ID = re.compile(
    r"(?:scope|case|packet|input|admin|freeze)_[0-9a-f]{64}\Z"
)
ISO8601 = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})\Z"
)

# These fields identify a comparison system, corpus row, or linked INDRA
# object.  Biomedical grounding IDs under db_refs remain visible because they
# are part of the scientific claim, not a link back to the comparison arm.
FORBIDDEN_MATERIAL_KEYS = frozenset(
    {
        "arm",
        "arm_id",
        "arm_family",
        "cost",
        "citation_ref",
        "content_source",
        "document_id",
        "doi",
        "evidence_hash",
        "execution_id",
        "execution_identity",
        "hash",
        "hashes",
        "id",
        "ids",
        "implementation",
        "matches_hash",
        "model",
        "model_id",
        "model_name",
        "panel",
        "panel_id",
        "pmid",
        "prediction",
        "probability",
        "probability_correct",
        "provider",
        "prior_uuids",
        "score",
        "source_hash",
        "source_id",
        "statement_id",
        "supported_by",
        "supports",
        "threshold",
        "text_refs",
        "url",
        "urls",
        "uuid",
        "uuids",
    }
)


class ErrorReviewError(ValueError):
    """An artifact cannot support the canonical human review workflow."""


def _fail(message: str) -> NoReturn:
    raise ErrorReviewError(message)


def _canonical_bytes(value: Any, *, pretty: bool = False) -> bytes:
    try:
        if pretty:
            return (
                json.dumps(
                    value,
                    sort_keys=True,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
        return canonical_json_bytes(value)
    except (ContractError, TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ErrorReviewError("value is not strict JSON") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read(path: Path) -> bytes:
    try:
        return stable_read(Path(path).resolve(), context="error-review input").payload
    except ContractError as exc:
        raise ErrorReviewError(str(exc)) from exc


def _json_payload(payload: bytes, *, context: str) -> Any:
    try:
        return strict_json_loads(payload, context=context)
    except ContractError as exc:
        raise ErrorReviewError(str(exc)) from exc


def load_json(path: Path) -> dict[str, Any]:
    value = _json_payload(_read(path), context=str(Path(path).resolve()))
    if not isinstance(value, dict):
        _fail(f"{path} must contain a JSON object")
    return value


def _jsonl_payload(payload: bytes, *, context: str) -> list[dict[str, Any]]:
    if not payload or not payload.endswith(b"\n"):
        _fail(f"{context} is empty or lacks a terminal LF")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(payload.splitlines(), 1):
        value = _json_payload(line, context=f"{context}:{number}")
        if not isinstance(value, dict):
            _fail(f"{context}:{number} must contain an object")
        rows.append(value)
    return rows


def _descriptor(path: Path, payload: bytes | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    payload = _read(path) if payload is None else payload
    return {"path": str(path), "sha256": _sha256(payload), "bytes": len(payload)}


def _public_commitment(value: Any) -> Any:
    """Remove local filesystem paths while retaining cryptographic provenance."""

    if isinstance(value, Mapping):
        return {
            key: _public_commitment(child)
            for key, child in value.items()
            if key != "path"
        }
    if isinstance(value, list):
        return [_public_commitment(child) for child in value]
    return value


def _check_descriptor(
    value: Any, *, owner: Path | None = None, context: str
) -> tuple[Path, bytes]:
    if not isinstance(value, Mapping):
        _fail(f"{context} must be a file descriptor")
    allowed = {"path", "sha256", "bytes", "rows"}
    if not {"path", "sha256"} <= set(value) or set(value) - allowed:
        _fail(f"{context} descriptor schema differs")
    declared = value.get("path")
    digest = value.get("sha256")
    if not isinstance(declared, str) or not declared:
        _fail(f"{context}.path must be non-empty text")
    if not isinstance(digest, str) or HEX64.fullmatch(digest) is None:
        _fail(f"{context}.sha256 must be a lowercase SHA-256")
    if "bytes" in value and (
        isinstance(value.get("bytes"), bool)
        or not isinstance(value.get("bytes"), int)
        or value["bytes"] < 0
    ):
        _fail(f"{context}.bytes must be a non-negative integer")
    if "rows" in value and (
        isinstance(value.get("rows"), bool)
        or not isinstance(value.get("rows"), int)
        or value["rows"] < 0
    ):
        _fail(f"{context}.rows must be a non-negative integer")
    path = Path(declared)
    if not path.is_absolute():
        if owner is None:
            _fail(f"{context}.path must be absolute")
        path = owner.resolve().parent / path
    path = path.resolve()
    payload = _read(path)
    if _sha256(payload) != digest:
        _fail(f"{context} digest differs")
    if "bytes" in value and value.get("bytes") != len(payload):
        _fail(f"{context} byte count differs")
    if "rows" in value and value.get("rows") != payload.count(b"\n"):
        _fail(f"{context} row count differs")
    return path, payload


def _check_large_descriptor(
    value: Any, *, owner: Path | None = None, context: str
) -> dict[str, Any]:
    """Verify a large descriptor without materializing the file in memory."""

    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "bytes"}:
        _fail(f"{context} descriptor schema differs")
    declared = value.get("path")
    digest = value.get("sha256")
    byte_count = value.get("bytes")
    if not isinstance(declared, str) or not declared:
        _fail(f"{context}.path must be non-empty text")
    if not isinstance(digest, str) or HEX64.fullmatch(digest) is None:
        _fail(f"{context}.sha256 must be a lowercase SHA-256")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        _fail(f"{context}.bytes must be a non-negative integer")
    path = Path(declared)
    if not path.is_absolute():
        if owner is None:
            _fail(f"{context}.path must be absolute")
        path = owner.resolve().parent / path
    path = Path(os.path.abspath(path))
    try:
        before_name = os.lstat(path)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ErrorReviewError(f"cannot open {context}: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before_name.st_mode)
            or before.st_nlink != 1
            or (before.st_dev, before.st_ino) != (before_name.st_dev, before_name.st_ino)
        ):
            _fail(f"{context} is not one stable regular file")
        hasher = hashlib.sha256()
        observed = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            observed += len(block)
            hasher.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_name = os.lstat(path)
    except OSError as exc:
        raise ErrorReviewError(f"{context} changed while hashing") from exc
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields) or any(
        getattr(before, field) != getattr(after_name, field) for field in stable_fields
    ):
        _fail(f"{context} changed while hashing")
    if observed != byte_count or before.st_size != byte_count:
        _fail(f"{context} byte count differs")
    if hasher.hexdigest() != digest:
        _fail(f"{context} digest differs")
    return {"path": str(path.resolve()), "sha256": digest, "bytes": byte_count}


def _key(value: bytes) -> bytes:
    if not isinstance(value, bytes) or len(value) < 32:
        _fail("blinding key must contain at least 32 bytes")
    return value


def load_blinding_key(path: Path) -> bytes:
    """Load a raw key, or a file containing exactly 64 hexadecimal digits."""

    path = Path(os.path.abspath(path))
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise ErrorReviewError(f"cannot stat blinding key at {path}") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        _fail("blinding key must be an owner-held regular file with no group/other access")
    raw = _read(path)
    stripped = raw.strip()
    if re.fullmatch(rb"[0-9a-fA-F]{64}", stripped):
        raw = bytes.fromhex(stripped.decode("ascii"))
    return _key(raw)


def generate_blinding_key(path: Path) -> None:
    """Create a new administrator-held 256-bit key with mode 0600."""

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise ErrorReviewError(f"refusing to replace existing blinding key at {path}") from exc
    try:
        payload = os.urandom(32).hex().encode("ascii") + b"\n"
        if os.write(descriptor, payload) != len(payload):
            _fail("short write while creating the blinding key")
        os.fsync(descriptor)
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(descriptor)


def _opaque(secret: bytes, prefix: str, value: Any) -> str:
    digest = hmac.new(
        _key(secret), prefix.encode("ascii") + b"\0" + _canonical_bytes(value), hashlib.sha256
    ).hexdigest()
    return f"{prefix}_{digest}"


def _identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        _fail(f"{context} must be non-empty text")
    return value


def _probability(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        _fail(f"{context} must be finite and inside [0, 1]")
    return result


def _label(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        _fail(f"{context} must be integer 0 or 1")
    return value


def _timestamp(value: Any, context: str) -> str:
    if not isinstance(value, str) or ISO8601.fullmatch(value) is None:
        _fail(f"{context} must be an ISO-8601 timestamp with a timezone")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ErrorReviewError(f"{context} is not a real timestamp") from exc
    if parsed.tzinfo is None:
        _fail(f"{context} must include a timezone")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write(path: Path, payload: bytes, *, private: bool = True) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read(path) == payload:
            os.chmod(path, 0o600 if private else 0o644)
            return
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600 if private else 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(value: Mapping[str, Any], path: Path) -> None:
    """Write an immutable canonical JSON artifact (idempotent for equal bytes)."""

    _atomic_write(path, _canonical_bytes(value, pretty=True))


def _validate_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path).resolve()
    payload = _read(path)
    if _sha256(payload) != CANONICAL_PROTOCOL_SHA256:
        _fail("error-review protocol is not the one canonical frozen contract")
    value = _json_payload(payload, context="error-review protocol")
    if (
        not isinstance(value, dict)
        or set(value) != {
            "artifact_kind", "blinding", "census", "error_definition",
            "frozen_at", "human_review", "provenance", "reporting",
        }
        or value.get("artifact_kind") != "indra_belief_error_review_contract"
    ):
        _fail("error-review protocol schema or artifact kind differs")
    _timestamp(value.get("frozen_at"), "protocol.frozen_at")
    error_definition = value.get("error_definition")
    if (
        not isinstance(error_definition, Mapping)
        or set(error_definition) != {
            "negative_class", "operator", "positive_class", "primary_threshold", "rule"
        }
        or error_definition.get("negative_class") != "incorrect_statement"
        or error_definition.get("positive_class") != "correct_statement"
        or error_definition.get("operator") != "greater_than_or_equal"
        or _probability(
            error_definition.get("primary_threshold"),
            "protocol.error_definition.primary_threshold",
        ) != 0.5
    ):
        _fail("error-review protocol threshold/class definition differs")
    human = value.get("human_review")
    if not isinstance(human, Mapping) or set(human) != {
        "comments", "dimensions", "classifications", "model_generated_adjudication",
        "resolver", "reviewers",
    }:
        _fail("error-review protocol human_review schema differs")
    classifications = human.get("classifications")
    if classifications != CLASSIFICATION_DEFINITIONS:
        _fail("protocol must define exactly the three outcome-blind claim classifications")
    for classification in CLASSIFICATIONS:
        _identifier(
            classifications[classification],
            f"protocol.human_review.classifications.{classification}",
        )
    if human.get("model_generated_adjudication") != "prohibited":
        _fail("protocol must prohibit model-generated adjudication")
    if human.get("reviewers") != "exactly two independent humans with complete case coverage":
        _fail("protocol must require exactly two complete independent human reviews")
    if human.get("resolver") != "a third human covers every reviewer disagreement and no agreement":
        _fail("protocol resolver rule differs")
    return value, _descriptor(path, payload)


def _default_dimensions() -> list[dict[str, str]]:
    return [
        {
            "dimension": "aggregation_conflict",
            "label": "Conflicting evidence aggregation",
            "definition": "Displayed evidence items materially conflict, making their aggregation decisive.",
        },
        {
            "dimension": "claim_scope_ambiguity",
            "label": "Claim scope ambiguity",
            "definition": "The assembled claim permits materially different reasonable scopes or interpretations.",
        },
        {
            "dimension": "context_mismatch",
            "label": "Biological context mismatch",
            "definition": "Species, cell, tissue, disease, location, or experimental context does not match the claim.",
        },
        {
            "dimension": "explicit_contradiction",
            "label": "Explicit contradiction",
            "definition": "Displayed evidence explicitly contradicts the assembled claim.",
        },
        {
            "dimension": "evidence_insufficient",
            "label": "Insufficient evidence",
            "definition": "Displayed evidence does not establish the assembled claim.",
        },
        {
            "dimension": "evidence_scope_ambiguity",
            "label": "Evidence scope ambiguity",
            "definition": "Displayed evidence is incomplete or ambiguous about the scope needed by the claim.",
        },
        {
            "dimension": "explicit_support",
            "label": "Explicit support",
            "definition": "Displayed evidence directly and unambiguously supports the assembled claim.",
        },
        {
            "dimension": "grounding_ambiguity",
            "label": "Entity grounding ambiguity",
            "definition": "Entity identity, family/member resolution, or grounding materially affects the judgment.",
        },
        {
            "dimension": "polarity_negation_ambiguity",
            "label": "Polarity or negation ambiguity",
            "definition": "Negation, hedging, direction, or polarity materially affects the judgment.",
        },
        {
            "dimension": "relation_type_ambiguity",
            "label": "Relation-type ambiguity",
            "definition": "Mapping the text to the assembled INDRA relation or event type is materially ambiguous.",
        },
        {
            "dimension": "source_reliability",
            "label": "Source reliability",
            "definition": "The extraction or database source's reliability materially affects a reasonable judgment.",
        },
        {
            "dimension": "taxonomy_gap",
            "label": "Taxonomy gap",
            "definition": "No existing dimension adequately describes the case; explain the proposed refinement in the comment.",
        },
    ]


def make_pilot_codebook(protocol_path: Path) -> dict[str, Any]:
    """Create the candidate codebook; this does not claim that a pilot occurred."""

    protocol, descriptor = _validate_protocol(protocol_path)
    classifications = protocol["human_review"]["classifications"]
    return {
        "artifact_kind": CODEBOOK_KIND,
        "status": "pilot",
        "protocol_sha256": descriptor["sha256"],
        "classification_rubric": {
            "ordered_steps": list(CLASSIFICATION_STEPS),
            **{
                classification: classifications[classification]
                for classification in CLASSIFICATIONS
            },
        },
        "dimensions": _default_dimensions(),
        "pilot_contract": dict(PILOT_CONTRACT),
        "frozen_at": None,
        "pilot_provenance": None,
        "human_freeze_attestation": None,
        "freeze_binding": None,
    }


def _validate_codebook(
    path: Path, *, protocol_sha256: str
) -> tuple[dict[str, Any], dict[str, Any], frozenset[str]]:
    path = Path(path).resolve()
    payload = _read(path)
    value = _json_payload(payload, context="error-review codebook")
    required = {
        "artifact_kind",
        "status",
        "protocol_sha256",
        "classification_rubric",
        "dimensions",
        "pilot_contract",
        "frozen_at",
        "pilot_provenance",
        "human_freeze_attestation",
        "freeze_binding",
    }
    if not isinstance(value, dict) or set(value) != required or value.get("artifact_kind") != CODEBOOK_KIND:
        _fail("codebook schema or artifact kind differs")
    if value.get("protocol_sha256") != protocol_sha256:
        _fail("codebook belongs to another protocol")
    status = value.get("status")
    if status not in {"pilot", "frozen"}:
        _fail("codebook status must be pilot or frozen")
    rubric = value.get("classification_rubric")
    if not isinstance(rubric, Mapping) or set(rubric) != {
        "ordered_steps", *CLASSIFICATIONS
    }:
        _fail("codebook classification rubric schema differs")
    if not isinstance(rubric.get("ordered_steps"), list) or not rubric["ordered_steps"]:
        _fail("codebook rubric must contain ordered steps")
    for index, step in enumerate(rubric["ordered_steps"]):
        _identifier(step, f"codebook.classification_rubric.ordered_steps[{index}]")
    for classification in CLASSIFICATIONS:
        _identifier(
            rubric.get(classification),
            f"codebook.classification_rubric.{classification}",
        )
    if rubric != {
        "ordered_steps": CLASSIFICATION_STEPS,
        **CLASSIFICATION_DEFINITIONS,
    }:
        _fail("codebook classification rubric differs from the canonical protocol")
    dimensions = value.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        _fail("codebook dimensions must be a non-empty array")
    identifiers: set[str] = set()
    for index, row in enumerate(dimensions):
        if not isinstance(row, Mapping) or set(row) != {"dimension", "label", "definition"}:
            _fail(f"codebook dimension {index} schema differs")
        dimension = row.get("dimension")
        if not isinstance(dimension, str) or DIMENSION_ID.fullmatch(dimension) is None:
            _fail(f"codebook dimension {index} has an invalid identifier")
        if dimension in identifiers:
            _fail(f"codebook repeats dimension {dimension!r}")
        identifiers.add(dimension)
        _identifier(row.get("label"), f"codebook dimension {index}.label")
        _identifier(row.get("definition"), f"codebook dimension {index}.definition")
    if value.get("pilot_contract") != PILOT_CONTRACT:
        _fail("codebook pilot contract differs from the canonical human workflow")
    if status == "pilot":
        if any(value.get(key) is not None for key in (
            "frozen_at", "pilot_provenance", "human_freeze_attestation",
            "freeze_binding",
        )):
            _fail("pilot codebook cannot claim frozen provenance")
    else:
        _timestamp(value.get("frozen_at"), "codebook.frozen_at")
        pilot_provenance = value.get("pilot_provenance")
        if not isinstance(pilot_provenance, Mapping) or set(pilot_provenance) != {
            "protocol",
            "source_pilot_codebook",
            "human_refined_candidate",
            "pilot_packet",
            "pilot_admin_manifest",
            "reviewer_ledgers",
            "reviewer_workbooks",
        }:
            _fail("frozen codebook lacks pilot provenance")
        if value.get("human_freeze_attestation") != FREEZE_ATTESTATION:
            _fail("frozen codebook lacks the human freeze attestation")
        if (
            not isinstance(value.get("freeze_binding"), str)
            or not value["freeze_binding"].startswith("freeze_")
            or OPAQUE_ID.fullmatch(value["freeze_binding"]) is None
        ):
            _fail("frozen codebook freeze binding is invalid")
        pilot_protocol = pilot_provenance.get("protocol")
        if not isinstance(pilot_protocol, Mapping) or pilot_protocol.get("sha256") != protocol_sha256:
            _fail("frozen codebook pilot protocol binding differs")
        _check_descriptor(pilot_protocol, context="frozen codebook protocol")
        pilot_path, _pilot_payload = _check_descriptor(
            pilot_provenance.get("source_pilot_codebook"),
            context="frozen codebook source pilot codebook",
        )
        candidate_path, _candidate_payload = _check_descriptor(
            pilot_provenance.get("human_refined_candidate"),
            context="frozen codebook human-refined candidate",
        )
        pilot_codebook, pilot_descriptor, pilot_dimensions = _validate_codebook(
            pilot_path, protocol_sha256=protocol_sha256
        )
        candidate, _candidate_descriptor, _candidate_dimensions = _validate_codebook(
            candidate_path, protocol_sha256=protocol_sha256
        )
        if pilot_codebook["status"] != "pilot" or candidate["status"] != "pilot":
            _fail("frozen codebook provenance does not bind pilot codebooks")
        if (
            value["classification_rubric"] != candidate["classification_rubric"]
            or value["dimensions"] != candidate["dimensions"]
            or value["pilot_contract"] != candidate["pilot_contract"]
        ):
            _fail("frozen codebook content differs from its human-refined candidate")
        packet_path, packet_payload = _check_descriptor(
            pilot_provenance.get("pilot_packet"), context="frozen codebook pilot packet"
        )
        packet = _json_payload(packet_payload, context="frozen codebook pilot packet")
        _validate_packet(
            packet,
            protocol_sha256=protocol_sha256,
            codebook_sha256=pilot_descriptor["sha256"],
        )
        if packet["review_phase"] != "pilot":
            _fail("frozen codebook provenance packet is not a pilot")
        _check_descriptor(
            pilot_provenance.get("pilot_admin_manifest"),
            context="frozen codebook pilot admin manifest",
        )
        workbook_descriptors = pilot_provenance.get("reviewer_workbooks")
        reviewer_descriptors = pilot_provenance.get("reviewer_ledgers")
        if (
            not isinstance(workbook_descriptors, list)
            or len(workbook_descriptors) != 2
            or
            not isinstance(reviewer_descriptors, list)
            or len(reviewer_descriptors) != 2
        ):
            _fail("frozen codebook reviewer provenance differs")
        for index, descriptor in enumerate(workbook_descriptors):
            _check_descriptor(
                descriptor, context=f"frozen codebook reviewer workbook {index}"
            )
        for index, descriptor in enumerate(reviewer_descriptors):
            _ledger_path, ledger_payload = _check_descriptor(
                descriptor, context=f"frozen codebook reviewer ledger {index}"
            )
            ledger = _json_payload(
                ledger_payload, context=f"frozen codebook reviewer ledger {index}"
            )
            if not isinstance(ledger, Mapping) or ledger.get("artifact_kind") != LEDGER_KIND:
                _fail("frozen codebook reviewer ledger artifact kind differs")
    return value, _descriptor(path, payload), frozenset(identifiers)


def _scrub_material(value: Any, *, forbidden_values: Iterable[str], context: str) -> Any:
    # UUID_TOKEN and HEX64_TOKEN reject the complete statement/execution identity
    # classes in one bounded regex pass.  Retain only non-token textual identities
    # (panel, arm, model, run) for substring checks; scanning every material string
    # against tens of thousands of already-covered hashes is redundant and quadratic.
    forbidden = frozenset(
        item
        for item in forbidden_values
        if item and UUID.fullmatch(item) is None and HEX64.fullmatch(item) is None
    )

    def scrub(item: Any, location: str) -> Any:
        if isinstance(item, Mapping):
            cleaned: dict[str, Any] = {}
            for raw_key, child in item.items():
                if not isinstance(raw_key, str):
                    _fail(f"{location} has a non-string key")
                normalized = raw_key.casefold()
                if (
                    normalized in FORBIDDEN_MATERIAL_KEYS
                    or normalized.endswith("_uuid")
                    or normalized.endswith("_uuids")
                    or normalized.endswith("_hash")
                    or normalized.endswith("_hashes")
                ):
                    continue
                cleaned[raw_key] = scrub(child, f"{location}.{raw_key}")
            return cleaned
        if isinstance(item, list):
            return [scrub(child, f"{location}[{index}]") for index, child in enumerate(item)]
        if isinstance(item, str):
            if UUID_TOKEN.search(item) or HEX64_TOKEN.search(item):
                _fail(f"{location} contains a linked UUID or hash value")
            if any(identity in item for identity in forbidden):
                _fail(f"{location} contains an input identity")
        return item

    result = scrub(value, context)
    _canonical_bytes(result)
    return result


def _rows_by_statement(
    rows: Sequence[Mapping[str, Any]], *, value_field: str, parser: Any, context: str
) -> tuple[list[str], dict[str, Any]]:
    order: list[str] = []
    indexed: dict[str, Any] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or "statement_id" not in row or value_field not in row:
            _fail(f"{context}[{index}] lacks statement_id or {value_field}")
        statement_id = _identifier(row.get("statement_id"), f"{context}[{index}].statement_id")
        if statement_id in indexed:
            _fail(f"{context} repeats statement_id {statement_id!r}")
        order.append(statement_id)
        indexed[statement_id] = parser(row.get(value_field), f"{context}[{index}].{value_field}")
    return order, indexed


def _execution_identity(map_row: Mapping[str, Any], *, served_model: str, workload: str) -> str:
    identity = {
        "model": served_model,
        "workload_mode": workload,
        "eligible_position": int(map_row["eligible_position"]),
        "paper_statement_hash": str(map_row["paper_statement_hash"]),
        "source_hash": str(map_row["source_hash"]),
        "evidence_json_sha256": str(map_row["evidence_json_sha256"]),
    }
    return _sha256(_canonical_bytes(identity))


def _bundle_scope(
    *,
    spec_path: Path,
    bundle_manifest_path: Path,
    panel_id: str,
    arm_id: str,
    protocol_path: Path,
) -> dict[str, Any]:
    """Load and cross-check the exact model/panel input universe."""

    spec_path = Path(spec_path).resolve()
    bundle_manifest_path = Path(bundle_manifest_path).resolve()
    protocol_path = Path(protocol_path).resolve()
    panel_id = _identifier(panel_id, "panel_id")
    arm_id = _identifier(arm_id, "arm_id")
    protocol, protocol_descriptor = _validate_protocol(protocol_path)
    spec_payload = _read(spec_path)
    bundle_payload = _read(bundle_manifest_path)
    spec = _json_payload(spec_payload, context="comparison spec")
    bundle = _json_payload(bundle_payload, context="LLM bundle manifest")
    if not isinstance(spec, dict) or spec.get("artifact_kind") != "indra_statement_belief_evaluation_spec":
        _fail("comparison spec artifact kind differs")
    if (
        not isinstance(bundle, dict)
        or set(bundle) != {"kind", "model_id", "run_id", "implementation", "panels"}
        or bundle.get("kind") != "llm_model_bundle"
    ):
        _fail("LLM bundle manifest is not the canonical schema")
    if bundle.get("model_id") != arm_id:
        _fail("bundle model_id differs from the requested comparison arm")
    run_id = _identifier(bundle.get("run_id"), "bundle.run_id")
    implementation = bundle.get("implementation")
    if not isinstance(implementation, Mapping) or set(implementation) != {
        "implementation",
        "implementation_digest",
        "training_data_sha256",
        "environment",
        "notes",
    }:
        _fail("bundle implementation is not the canonical schema")
    notes = implementation.get("notes")
    canonical_note_keys = {
        "aggregation",
        "dedup",
        "implementation_components",
        "inputs",
        "priors_sha256",
        "provider_model_id",
        "reader_profile",
        "reader_sources",
        "served_model",
        "true_reader_reaggregated_from_pair_measurements",
        "workload",
    }
    if not isinstance(notes, Mapping) or set(notes) != canonical_note_keys:
        _fail("bundle implementation notes are not the canonical schema")
    if notes.get("dedup") is not True or notes.get("true_reader_reaggregated_from_pair_measurements") is not True:
        _fail("bundle does not attest exact-pair deduplication and true reader reaggregation")
    _identifier(notes.get("aggregation"), "bundle notes aggregation")
    reader_sources = notes.get("reader_sources")
    if (
        not isinstance(reader_sources, list)
        or any(not isinstance(source, str) for source in reader_sources)
        or frozenset(source.casefold() for source in reader_sources) != READER_SOURCES
        or len(reader_sources) != len(READER_SOURCES)
    ):
        _fail("bundle reader-source definition differs from the canonical five-reader set")

    substrates = spec.get("substrates")
    if not isinstance(substrates, list):
        _fail("comparison spec substrates must be an array")
    selected_panels = [
        row for row in substrates
        if isinstance(row, Mapping) and row.get("substrate_id") == panel_id
    ]
    if len(selected_panels) != 1:
        _fail(f"comparison spec must contain exactly one panel {panel_id!r}")
    spec_panel = selected_panels[0]
    arms = spec_panel.get("arms")
    if not isinstance(arms, list):
        _fail("comparison panel arms must be an array")
    selected_arms = [
        row for row in arms
        if isinstance(row, Mapping) and row.get("arm_id") == arm_id
    ]
    if len(selected_arms) != 1:
        _fail(f"comparison panel must contain exactly one arm {arm_id!r}")
    spec_arm = selected_arms[0]
    if spec_arm.get("family") != "llm":
        _fail("error review requires an evaluated LLM arm")

    bundle_panels = bundle.get("panels")
    if (
        not isinstance(bundle_panels, Mapping)
        or set(bundle_panels) != {"paper_all_source", "paper_readers"}
        or panel_id not in bundle_panels
    ):
        _fail("bundle does not contain the requested panel")
    bundle_panel = bundle_panels[panel_id]
    if (
        not isinstance(bundle_panel, Mapping)
        or set(bundle_panel) != {"prediction_unit", "substrate_id", "predictions", "cost"}
        or bundle_panel.get("substrate_id") != panel_id
        or bundle_panel.get("prediction_unit") != "assembled_statement"
    ):
        _fail("bundle panel schema or identity differs")
    if not isinstance(spec_arm.get("predictions"), Mapping) or set(spec_arm["predictions"]) != {
        "path", "sha256"
    }:
        _fail("spec arm prediction descriptor schema differs")
    if not isinstance(bundle_panel.get("predictions"), Mapping) or set(bundle_panel["predictions"]) != {
        "path", "sha256", "bytes", "rows"
    }:
        _fail("bundle panel prediction descriptor schema differs")

    spec_prediction_path, spec_prediction_payload = _check_descriptor(
        spec_arm.get("predictions"), owner=spec_path, context="spec arm predictions"
    )
    bundle_prediction_path, bundle_prediction_payload = _check_descriptor(
        bundle_panel.get("predictions"),
        owner=bundle_manifest_path,
        context="bundle panel predictions",
    )
    if (
        spec_prediction_path != bundle_prediction_path
        or spec_prediction_payload != bundle_prediction_payload
    ):
        _fail("spec arm predictions are not the selected bundle predictions")

    spec_cost = spec_arm.get("cost")
    if (
        not isinstance(spec_cost, Mapping)
        or set(spec_cost) != {
            "accounting", "additive_across_panels", "basis",
            "cost_comparability_id", "counterfactual_run_cost", "path",
            "price_date", "price_source", "pricing", "projection",
            "record_type", "sha256", "shared_run_id", "status", "view_id",
        }
        or spec_cost.get("status") != "ledger"
    ):
        _fail("spec LLM arm must have a bound execution ledger")
    bundle_cost = bundle_panel.get("cost")
    if not isinstance(bundle_cost, Mapping) or set(bundle_cost) != {
        "accounting", "additive_across_panels", "basis", "bytes",
        "cost_comparability_id", "counterfactual_run_cost", "path",
        "price_date", "price_source", "pricing", "projection", "record_type",
        "rows", "sha256", "shared_run_id", "status", "view_id",
    }:
        _fail("bundle panel execution-ledger descriptor schema differs")
    spec_cost_path, spec_cost_payload = _check_descriptor(
        {"path": spec_cost["path"], "sha256": spec_cost["sha256"]},
        owner=spec_path,
        context="spec arm cost",
    )
    bundle_cost_path, bundle_cost_payload = _check_descriptor(
        {
            field: bundle_cost[field]
            for field in ("path", "sha256", "bytes", "rows")
        },
        owner=bundle_manifest_path,
        context="bundle panel cost",
    )
    if spec_cost_path != bundle_cost_path or spec_cost_payload != bundle_cost_payload:
        _fail("spec arm cost is not the selected bundle execution ledger")
    cost_contract_fields = {
        "accounting",
        "additive_across_panels",
        "basis",
        "cost_comparability_id",
        "counterfactual_run_cost",
        "price_date",
        "price_source",
        "pricing",
        "projection",
        "record_type",
        "shared_run_id",
        "status",
        "view_id",
    }
    if any(spec_cost[field] != bundle_cost[field] for field in cost_contract_fields):
        _fail("spec and bundle cost contracts differ")

    threshold = spec_arm.get("threshold")
    if (
        not isinstance(threshold, Mapping)
        or set(threshold) != {
            "frozen_at", "operator", "source_path", "source_sha256", "status", "value"
        }
        or threshold.get("status") != "available"
        or threshold.get("operator") != protocol["error_definition"]["operator"]
        or threshold.get("frozen_at") != protocol["frozen_at"]
    ):
        _fail("selected arm lacks the canonical threshold")
    threshold_value = _probability(threshold.get("value"), "spec arm threshold.value")
    if threshold_value != protocol["error_definition"]["primary_threshold"]:
        _fail("selected arm threshold value differs from the frozen review protocol")
    threshold_source = threshold.get("source_path")
    if not isinstance(threshold_source, str) or not threshold_source:
        _fail("selected arm threshold lacks its protocol path")
    resolved_threshold_source = (
        Path(threshold_source)
        if Path(threshold_source).is_absolute()
        else spec_path.parent / threshold_source
    ).resolve()
    if resolved_threshold_source != protocol_path:
        _fail("selected arm threshold points to another review protocol")
    protocol_payload = _read(protocol_path)
    if threshold.get("source_sha256") != protocol_descriptor["sha256"]:
        _fail("selected arm threshold protocol digest differs")

    if not isinstance(spec_panel.get("gold"), Mapping) or set(spec_panel["gold"]) != {
        "path", "sha256"
    }:
        _fail("spec panel gold descriptor schema differs")
    gold_path, gold_payload = _check_descriptor(
        spec_panel.get("gold"), owner=spec_path, context="spec panel gold"
    )
    gold_rows = _jsonl_payload(gold_payload, context="panel gold")
    prediction_rows = _jsonl_payload(bundle_prediction_payload, context="panel predictions")
    for index, row in enumerate(gold_rows):
        if set(row) != {"statement_id", "label", "fold_id"}:
            _fail(f"panel gold row {index} schema differs")
        fold = row.get("fold_id")
        if isinstance(fold, bool) or not isinstance(fold, int) or not 0 <= fold <= 9:
            _fail(f"panel gold row {index} fold_id differs")
    for index, row in enumerate(prediction_rows):
        if set(row) != {"statement_id", "probability_correct"}:
            _fail(f"panel prediction row {index} schema differs")
    gold_order, gold = _rows_by_statement(
        gold_rows, value_field="label", parser=_label, context="gold"
    )
    prediction_order, predictions = _rows_by_statement(
        prediction_rows,
        value_field="probability_correct",
        parser=_probability,
        context="predictions",
    )
    if not gold_order or gold_order != prediction_order:
        _fail("bundle prediction coverage differs from exact panel gold")

    inputs = notes.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "aggregation_config", "execution_map", "pricing_config", "raw_attempts",
        "spend_ledger", "statements",
    }:
        _fail("bundle implementation inputs differ from the canonical input set")
    for name, descriptor in inputs.items():
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "path", "sha256", "bytes"
        }:
            _fail(f"bundle input {name!r} descriptor schema differs")
    raw_attempts_descriptor = _check_large_descriptor(
        inputs.get("raw_attempts"),
        owner=bundle_manifest_path,
        context="bundle raw attempts",
    )
    spend_ledger_descriptor = _check_large_descriptor(
        inputs.get("spend_ledger"),
        owner=bundle_manifest_path,
        context="bundle spend ledger",
    )
    statements_path, statements_payload = _check_descriptor(
        inputs.get("statements"), owner=bundle_manifest_path, context="bundle statements"
    )
    map_path, map_payload = _check_descriptor(
        inputs.get("execution_map"), owner=bundle_manifest_path, context="bundle execution map"
    )
    aggregation_path, aggregation_payload = _check_descriptor(
        inputs.get("aggregation_config"),
        owner=bundle_manifest_path,
        context="bundle aggregation config",
    )
    pricing_path, pricing_payload = _check_descriptor(
        inputs.get("pricing_config"),
        owner=bundle_manifest_path,
        context="bundle pricing config",
    )
    aggregation = _json_payload(aggregation_payload, context="bundle aggregation config")
    pricing = _json_payload(pricing_payload, context="bundle pricing config")
    if (
        not isinstance(aggregation, Mapping)
        or set(aggregation) != {"aggregation", "kind", "priors", "reader_profile"}
        or aggregation.get("kind") != "statement_belief_aggregation"
        or aggregation.get("aggregation") != notes.get("aggregation")
        or _sha256(_canonical_bytes(aggregation.get("priors")))
        != notes.get("priors_sha256")
        or aggregation.get("reader_profile") != notes.get("reader_profile")
    ):
        _fail("bundle aggregation config differs from implementation notes")
    pricing_fields = {
        "cost_comparability_id", "currency", "kind", "provider", "pricing_mode",
        "region", "resolved_service_tier", "retrieved_on", "service_tier_request",
        "source_url", "tariffs", "unit",
    }
    provider_model_id = _identifier(
        notes.get("provider_model_id"), "bundle notes provider_model_id"
    )
    tariffs = pricing.get("tariffs") if isinstance(pricing, Mapping) else None
    selected_tariff = tariffs.get(provider_model_id) if isinstance(tariffs, Mapping) else None
    expected_panel_pricing = (
        {
            "cost_comparability_id": pricing.get("cost_comparability_id"),
            "currency": pricing.get("currency"),
            "provider": pricing.get("provider"),
            "provider_model_id": provider_model_id,
            "pricing_mode": pricing.get("pricing_mode"),
            "region": pricing.get("region"),
            "resolved_service_tier": pricing.get("resolved_service_tier"),
            "retrieved_on": pricing.get("retrieved_on"),
            "service_tier_request": pricing.get("service_tier_request"),
            "source_url": pricing.get("source_url"),
            "tariff": selected_tariff,
            "unit": pricing.get("unit"),
        }
        if isinstance(pricing, Mapping)
        else None
    )
    if (
        not isinstance(pricing, Mapping)
        or set(pricing) != pricing_fields
        or pricing.get("kind") != "provider_token_pricing"
        or not isinstance(selected_tariff, Mapping)
        or set(selected_tariff) != {
            "input_usd_per_million", "output_usd_per_million", "pricing_basis"
        }
        or pricing.get("cost_comparability_id")
        != bundle_cost.get("cost_comparability_id")
        or pricing.get("source_url") != bundle_cost.get("price_source")
        or pricing.get("retrieved_on") != bundle_cost.get("price_date")
        or bundle_cost.get("pricing") != expected_panel_pricing
    ):
        _fail("bundle pricing config differs from the panel cost contract")
    statements = _json_payload(statements_payload, context="bundle statements")
    map_rows = _jsonl_payload(map_payload, context="bundle execution map")
    if not isinstance(statements, list) or not statements:
        _fail("bundle statements must be a non-empty array")
    served_model = _identifier(notes.get("served_model"), "bundle notes served_model")
    workload = _identifier(notes.get("workload"), "bundle notes workload")

    statement_ids: list[str] = []
    expected_keys: set[tuple[int, int]] = set()
    for statement_index, statement in enumerate(statements):
        if not isinstance(statement, Mapping):
            _fail(f"bundle statement {statement_index} is not an object")
        statement_id = _identifier(statement.get("id"), f"statement {statement_index}.id")
        if UUID.fullmatch(statement_id) is None:
            _fail(f"bundle statement {statement_index}.id is not a UUID")
        matches_hash = statement.get("matches_hash")
        if not isinstance(matches_hash, str) or not matches_hash.lstrip("-").isdecimal():
            _fail(f"bundle statement {statement_index}.matches_hash is not decimal text")
        _identifier(statement.get("type"), f"bundle statement {statement_index}.type")
        evidence = statement.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            _fail(f"bundle statement {statement_index}.evidence must be non-empty")
        for evidence_index, item in enumerate(evidence):
            if not isinstance(item, Mapping):
                _fail(f"bundle evidence {statement_index}/{evidence_index} is not an object")
            _identifier(
                item.get("source_api"),
                f"bundle evidence {statement_index}/{evidence_index}.source_api",
            )
            source_hash = item.get("source_hash")
            if isinstance(source_hash, bool) or not isinstance(source_hash, int):
                _fail(f"bundle evidence {statement_index}/{evidence_index}.source_hash is not integer")
            if item.get("text") is not None and not isinstance(item.get("text"), str):
                _fail(f"bundle evidence {statement_index}/{evidence_index}.text differs")
        statement_ids.append(statement_id)
        expected_keys.update((statement_index, index) for index in range(len(evidence)))
    if len(set(statement_ids)) != len(statement_ids):
        _fail("bundle statement IDs repeat")

    map_schema = {
        "canonical_corpus_row_index",
        "canonical_for_unique_pair",
        "eligible_position",
        "evidence_json_sha256",
        "evidence_position",
        "main_prompt_base_sha256",
        "new_evidence_i",
        "new_stmt_i",
        "pair_multiplicity",
        "paper_statement_hash",
        "relation_prompt_sha256",
        "route",
        "source_api",
        "source_hash",
        "statement_type",
        "variant_ordinal",
    }
    pairs_by_execution: dict[str, tuple[str, Mapping[str, Any], Mapping[str, Any]]] = {}
    seen_keys: set[tuple[int, int]] = set()
    for index, map_row in enumerate(map_rows):
        if not isinstance(map_row, Mapping) or set(map_row) != map_schema:
            _fail(f"execution map row {index} schema differs")
        statement_index = map_row.get("new_stmt_i")
        evidence_index = map_row.get("new_evidence_i")
        key = (statement_index, evidence_index)
        if (
            isinstance(statement_index, bool)
            or not isinstance(statement_index, int)
            or isinstance(evidence_index, bool)
            or not isinstance(evidence_index, int)
            or key not in expected_keys
            or key in seen_keys
            or map_row.get("eligible_position") != statement_index
        ):
            _fail(f"execution map row {index} has a foreign or duplicate corpus key")
        seen_keys.add(key)
        statement = statements[statement_index]
        evidence = statement["evidence"][evidence_index]
        if not isinstance(evidence, Mapping):
            _fail(f"statement evidence {statement_index}/{evidence_index} is not an object")
        integer_fields = (
            "canonical_corpus_row_index", "eligible_position", "evidence_position",
            "new_evidence_i", "new_stmt_i", "pair_multiplicity", "variant_ordinal",
        )
        if any(
            isinstance(map_row.get(field), bool)
            or not isinstance(map_row.get(field), int)
            or map_row[field] < (1 if field == "pair_multiplicity" else 0)
            for field in integer_fields
        ):
            _fail(f"execution map row {index} integer fields differ")
        relation_digest = map_row.get("relation_prompt_sha256")
        if (
            map_row.get("canonical_for_unique_pair") is not True
            or map_row.get("variant_ordinal") != 0
            or map_row.get("route") not in {
                "plain", "tool", "deterministic_mismatch",
                "deterministic_pseudogene", "no_text",
            }
            or not isinstance(map_row.get("paper_statement_hash"), str)
            or not map_row["paper_statement_hash"].lstrip("-").isdecimal()
            or not isinstance(map_row.get("source_hash"), str)
            or not map_row["source_hash"].lstrip("-").isdecimal()
            or not isinstance(map_row.get("evidence_json_sha256"), str)
            or HEX64.fullmatch(map_row["evidence_json_sha256"]) is None
            or not isinstance(map_row.get("main_prompt_base_sha256"), str)
            or HEX64.fullmatch(map_row["main_prompt_base_sha256"]) is None
            or (
                relation_digest is not None
                and (not isinstance(relation_digest, str) or HEX64.fullmatch(relation_digest) is None)
            )
            or map_row.get("statement_type") != statement.get("type")
            or not isinstance(map_row.get("source_api"), str)
            or not map_row["source_api"]
            or map_row.get("evidence_json_sha256") != _sha256(_canonical_bytes(evidence))
            or str(map_row.get("source_hash")) != str(evidence.get("source_hash"))
            or str(map_row.get("source_api") or "").casefold()
            != str(evidence.get("source_api") or "").casefold()
            or str(map_row.get("paper_statement_hash")) != str(statement.get("matches_hash"))
        ):
            _fail(f"execution map row {index} differs from its statement evidence")
        execution_id = _execution_identity(
            map_row, served_model=served_model, workload=workload
        )
        if execution_id in pairs_by_execution:
            _fail("execution identities repeat")
        pairs_by_execution[execution_id] = (
            statement_ids[statement_index], statement, evidence
        )
    if seen_keys != expected_keys:
        _fail("execution map does not cover the exact statement-evidence universe")

    cost_rows = _jsonl_payload(bundle_cost_payload, context="panel execution ledger")
    selected_execution_ids: list[str] = []
    selected_execution_id_set: set[str] = set()
    execution_ids_by_statement: defaultdict[str, list[str]] = defaultdict(list)
    evidence_by_statement: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(cost_rows):
        if row.get("record_type") != "evidence_execution":
            _fail(f"panel execution ledger row {index} has another record type")
        execution_id = row.get("execution_identity")
        statement_id = row.get("statement_id")
        if (
            not isinstance(execution_id, str)
            or HEX64.fullmatch(execution_id) is None
            or execution_id in selected_execution_id_set
            or execution_id not in pairs_by_execution
        ):
            _fail(f"panel execution ledger row {index} has a foreign or duplicate identity")
        expected_statement_id, _statement, evidence = pairs_by_execution[execution_id]
        if statement_id != expected_statement_id:
            _fail(f"panel execution ledger row {index} statement binding differs")
        selected_execution_ids.append(execution_id)
        selected_execution_id_set.add(execution_id)
        execution_ids_by_statement[statement_id].append(execution_id)
        evidence_by_statement[statement_id].append(evidence)
    if bundle_panel["cost"].get("rows") != len(cost_rows):
        _fail("bundle panel cost row count differs")
    ordered_all = list(pairs_by_execution)
    ordered_readers = [
        execution_id
        for execution_id, (_statement_id, _statement, evidence) in pairs_by_execution.items()
        if str(evidence.get("source_api") or "").casefold() in READER_SOURCES
    ]
    expected_projection = ordered_all if panel_id == "paper_all_source" else ordered_readers
    if selected_execution_ids != expected_projection:
        _fail("panel execution ledger is not the exact ordered execution projection")
    if set(gold_order) != set(evidence_by_statement):
        _fail("panel gold does not equal statements represented by exact execution identities")

    statement_by_id = {
        statement_id: statements[index] for index, statement_id in enumerate(statement_ids)
    }
    forbidden_values = {
        panel_id,
        arm_id,
        str(bundle.get("model_id")),
        run_id,
        *statement_ids,
        *pairs_by_execution,
    }
    material_by_statement: dict[str, dict[str, Any]] = {}
    for statement_id in gold_order:
        statement = statement_by_id.get(statement_id)
        if statement is None:
            _fail(f"panel statement {statement_id!r} is absent from the bundle corpus")
        claim = {key: child for key, child in statement.items() if key != "evidence"}
        material_by_statement[statement_id] = _scrub_material(
            {"claim": claim, "evidence": evidence_by_statement[statement_id]},
            forbidden_values=forbidden_values,
            context=f"material[{statement_id!r}]",
        )

    provenance = {
        "panel_id": panel_id,
        "arm_id": arm_id,
        "model_id": bundle["model_id"],
        "run_id": run_id,
        "threshold": threshold_value,
        "files": {
            "spec": _descriptor(spec_path, spec_payload),
            "bundle_manifest": _descriptor(bundle_manifest_path, bundle_payload),
            "protocol": _descriptor(protocol_path, protocol_payload),
            "gold": _descriptor(gold_path, gold_payload),
            "predictions": _descriptor(bundle_prediction_path, bundle_prediction_payload),
            "execution_ledger": _descriptor(bundle_cost_path, bundle_cost_payload),
            "statements": _descriptor(statements_path, statements_payload),
            "execution_map": _descriptor(map_path, map_payload),
            "aggregation_config": _descriptor(aggregation_path, aggregation_payload),
            "pricing_config": _descriptor(pricing_path, pricing_payload),
            "raw_attempts": raw_attempts_descriptor,
            "spend_ledger": spend_ledger_descriptor,
        },
        "ordered_gold_statement_id_sha256": _sha256(_canonical_bytes(gold_order)),
        "selected_execution_projection_sha256": _sha256(
            b"".join(
                _canonical_bytes(
                    {
                        "execution_identity": execution_id,
                        "statement_id": pairs_by_execution[execution_id][0],
                    }
                ) + b"\n"
                for execution_id in selected_execution_ids
            )
        ),
        "selected_execution_count": len(selected_execution_ids),
    }
    return {
        "gold_order": gold_order,
        "gold": gold,
        "predictions": predictions,
        "material": material_by_statement,
        "execution_ids_by_statement": dict(execution_ids_by_statement),
        "provenance": provenance,
    }


def _select_pilot_cases(
    errors: Sequence[dict[str, Any]], *, count: int
) -> list[dict[str, Any]]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        _fail("pilot_case_count must be an integer of at least 2")
    if count % 2:
        _fail("pilot_case_count must be even for exact direction balance")
    if count > len(errors):
        _fail("pilot_case_count exceeds the threshold-error census")
    by_direction = {
        direction: sorted(
            (row for row in errors if row["_error_type"] == direction),
            key=lambda row: row["case_id"],
        )
        for direction in ("false_positive", "false_negative")
    }
    selected: list[dict[str, Any]] = []
    target_each = count // 2
    if any(len(rows) < target_each for rows in by_direction.values()):
        _fail("pilot_case_count cannot be balanced across both error directions")
    for direction in ("false_positive", "false_negative"):
        selected.extend(by_direction[direction][:target_each])
    return sorted(selected, key=lambda row: row["case_id"])


def prepare_review_artifacts(
    *,
    spec_path: Path,
    bundle_manifest_path: Path,
    panel_id: str,
    arm_id: str,
    protocol_path: Path,
    codebook_path: Path,
    blinding_key: bytes,
    reviewer_output_dir: Path,
    admin_output_dir: Path,
    pilot_case_count: int | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Publish one outcome-blind packet and its separate private admin manifest."""

    secret = _key(blinding_key)
    reviewer_output_dir = Path(reviewer_output_dir).resolve()
    admin_output_dir = Path(admin_output_dir).resolve()
    if (
        reviewer_output_dir == admin_output_dir
        or reviewer_output_dir in admin_output_dir.parents
        or admin_output_dir in reviewer_output_dir.parents
    ):
        _fail(
            "reviewer and administrator outputs must use disjoint, non-nested directories"
        )
    _protocol, protocol_descriptor = _validate_protocol(protocol_path)
    codebook, codebook_descriptor, _dimensions = _validate_codebook(
        codebook_path, protocol_sha256=protocol_descriptor["sha256"]
    )
    if codebook["status"] == "frozen":
        _validate_frozen_codebook_authenticity(codebook, secret=secret)
    phase = "pilot" if codebook["status"] == "pilot" else "full"
    if phase == "pilot" and pilot_case_count is None:
        _fail("pilot codebook requires pilot_case_count")
    if phase == "full" and pilot_case_count is not None:
        _fail("full review forbids pilot sampling")
    scope = _bundle_scope(
        spec_path=spec_path,
        bundle_manifest_path=bundle_manifest_path,
        panel_id=panel_id,
        arm_id=arm_id,
        protocol_path=protocol_path,
    )
    provenance = scope["provenance"]
    if provenance["files"]["protocol"]["sha256"] != protocol_descriptor["sha256"]:
        _fail("bundle scope and supplied protocol differ")
    input_binding = _opaque(secret, "input", provenance)
    scope_id = _opaque(secret, "scope", input_binding)
    threshold = provenance["threshold"]

    errors: list[dict[str, Any]] = []
    admin_rows: dict[str, dict[str, Any]] = {}
    directions = {"false_positive": 0, "false_negative": 0}
    for statement_id in scope["gold_order"]:
        reference = scope["gold"][statement_id]
        probability = scope["predictions"][statement_id]
        predicted = int(probability >= threshold)
        if predicted == reference:
            continue
        direction = "false_positive" if predicted == 1 else "false_negative"
        directions[direction] += 1
        case_id = _opaque(secret, "case", [input_binding, statement_id])
        errors.append(
            {
                "case_id": case_id,
                "material": scope["material"][statement_id],
                "_error_type": direction,
            }
        )
        admin_rows[case_id] = {
            "case_id": case_id,
            "statement_id": statement_id,
            "reference_label": reference,
            "probability_correct": probability,
            "error_type": direction,
            "execution_identities": scope["execution_ids_by_statement"][statement_id],
        }
    selected = (
        _select_pilot_cases(errors, count=pilot_case_count)
        if phase == "pilot" and pilot_case_count is not None
        else errors
    )
    selected_public = [
        {
            "case_id": row["case_id"],
            "material": row["material"],
        }
        for row in selected
    ]
    packet_body: dict[str, Any] = {
        "artifact_kind": PACKET_KIND,
        "review_phase": phase,
        "protocol_sha256": protocol_descriptor["sha256"],
        "codebook_sha256": codebook_descriptor["sha256"],
        "input_binding": input_binding,
        "scope_id": scope_id,
        "evaluated_statement_count": len(scope["gold_order"]),
        "threshold_error_count": len(errors),
        "review_case_count": len(selected),
        "cases": selected_public,
    }
    packet_body["packet_id"] = _opaque(secret, "packet", packet_body)
    packet_payload = _canonical_bytes(packet_body, pretty=True)
    packet_path = reviewer_output_dir / f"{packet_body['packet_id']}.json"
    packet_descriptor = _descriptor(packet_path, packet_payload)
    created = _timestamp(created_at, "created_at") if created_at is not None else _now()
    admin_body: dict[str, Any] = {
        "artifact_kind": ADMIN_KIND,
        "created_at": created,
        "input_binding": input_binding,
        "scope_id": scope_id,
        "provenance": provenance,
        "protocol": protocol_descriptor,
        "codebook": codebook_descriptor,
        "packet": packet_descriptor,
        "packet_id": packet_body["packet_id"],
        "threshold_error_census": {
            "count": len(errors),
            "false_positive": directions["false_positive"],
            "false_negative": directions["false_negative"],
        },
        "threshold_error_statement_ids": [
            row["statement_id"] for row in admin_rows.values()
        ],
        "case_mapping": [
            admin_rows[row["case_id"]] for row in selected
        ],
    }
    admin_body["admin_id"] = _opaque(secret, "admin", admin_body)
    admin_payload = _canonical_bytes(admin_body, pretty=True)
    admin_path = admin_output_dir / f"{admin_body['admin_id']}.json"
    _atomic_write(packet_path, packet_payload, private=False)
    # The private manifest is the commit marker.  Recheck every committed
    # input after publishing the identity-free packet and before publishing
    # the manifest that makes the packet scientifically usable.
    for name, descriptor in provenance["files"].items():
        if name in {"raw_attempts", "spend_ledger"}:
            _check_large_descriptor(
                descriptor, context=f"review preparation commit {name}"
            )
        else:
            _check_descriptor(descriptor, context=f"review preparation commit {name}")
    _check_descriptor(protocol_descriptor, context="review preparation commit protocol")
    _check_descriptor(codebook_descriptor, context="review preparation commit codebook")
    _atomic_write(admin_path, admin_payload)
    return {
        "packet": packet_path,
        "packet_sha256": _sha256(packet_payload),
        "admin_manifest": admin_path,
        "admin_manifest_sha256": _sha256(admin_payload),
        "review_phase": phase,
        "review_case_count": len(selected),
        "threshold_error_count": len(errors),
    }


def _validate_packet(
    value: Any,
    *,
    protocol_sha256: str,
    codebook_sha256: str,
    secret: bytes | None = None,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    required = {
        "artifact_kind",
        "review_phase",
        "protocol_sha256",
        "codebook_sha256",
        "input_binding",
        "scope_id",
        "evaluated_statement_count",
        "threshold_error_count",
        "review_case_count",
        "cases",
        "packet_id",
    }
    if not isinstance(value, Mapping) or set(value) != required or value.get("artifact_kind") != PACKET_KIND:
        _fail("review packet schema or artifact kind differs")
    if value.get("review_phase") not in {"pilot", "full"}:
        _fail("review packet phase differs")
    if value.get("protocol_sha256") != protocol_sha256:
        _fail("review packet belongs to another protocol")
    if value.get("codebook_sha256") != codebook_sha256:
        _fail("review packet belongs to another codebook")
    for field in ("input_binding", "scope_id", "packet_id"):
        if not isinstance(value.get(field), str) or OPAQUE_ID.fullmatch(value[field]) is None:
            _fail(f"review packet {field} is invalid")
    unsigned = dict(value)
    packet_id = unsigned.pop("packet_id")
    if secret is not None and packet_id != _opaque(secret, "packet", unsigned):
        _fail("review packet binding is invalid")
    counts = (
        value.get("evaluated_statement_count"),
        value.get("review_case_count"),
    )
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in counts):
        _fail("review packet counts must be non-negative integers")
    threshold_error_count = value.get("threshold_error_count")
    if (
        isinstance(threshold_error_count, bool)
        or not isinstance(threshold_error_count, int)
        or threshold_error_count < 0
    ):
        _fail("review packet threshold-error count differs")
    if threshold_error_count > counts[0]:
        _fail("review packet threshold errors exceed evaluated statements")
    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) != counts[1]:
        _fail("review packet case count differs")
    if value["review_phase"] == "full" and len(cases) != threshold_error_count:
        _fail("full-review packet does not cover every threshold error")
    seen: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or set(case) != {"case_id", "material"}:
            _fail(f"review packet case {index} schema differs")
        case_id = case.get("case_id")
        if (
            not isinstance(case_id, str)
            or OPAQUE_ID.fullmatch(case_id) is None
            or not case_id.startswith("case_")
            or case_id in seen
        ):
            _fail(f"review packet case {index} ID is invalid or duplicate")
        seen.add(case_id)
        scrubbed = _scrub_material(
            case.get("material"), forbidden_values=(), context=f"packet case {index}.material"
        )
        if scrubbed != case.get("material"):
            _fail(f"review packet case {index} contains an identity-bearing material key")
    return packet_id, cases


def _decision(
    value: Any,
    *,
    context: str,
    case_ids: set[str],
    dimensions: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "case_id", "classification", "dimensions", "comment"
    }:
        _fail(
            f"{context} must contain exactly case_id, classification, dimensions, and comment"
        )
    case_id = value.get("case_id")
    if not isinstance(case_id, str) or case_id not in case_ids:
        _fail(f"{context}.case_id is outside the assigned workload")
    classification = value.get("classification")
    if classification not in CLASSIFICATIONS:
        _fail(f"{context}.classification differs from the frozen rubric")
    raw_dimensions = value.get("dimensions")
    if not isinstance(raw_dimensions, list) or not raw_dimensions:
        _fail(f"{context}.dimensions must contain at least one frozen dimension")
    parsed_dimensions: list[str] = []
    for dimension in raw_dimensions:
        if not isinstance(dimension, str) or dimension not in dimensions:
            _fail(f"{context}.dimensions contains an identifier outside the codebook")
        if dimension in parsed_dimensions:
            _fail(f"{context}.dimensions contains a duplicate")
        parsed_dimensions.append(dimension)
    comment = value.get("comment")
    if comment is not None and (
        not isinstance(comment, str)
        or not comment.strip()
        or comment != comment.strip()
        or len(comment) > 4_000
        or "\x00" in comment
    ):
        _fail(
            f"{context}.comment must be null or trimmed substantive text of at most 4,000 characters"
        )
    if "taxonomy_gap" in parsed_dimensions and (not isinstance(comment, str) or not comment.strip()):
        _fail(f"{context}.comment is required for taxonomy_gap")
    return {
        "case_id": case_id,
        "classification": classification,
        "dimensions": sorted(parsed_dimensions),
        "comment": comment,
    }


def _decision_signature(value: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return fields that require human resolution when reviewers differ."""

    taxonomy_comment = (
        str(value.get("comment") or "").strip()
        if "taxonomy_gap" in value.get("dimensions", ())
        else None
    )
    return (
        value.get("classification"),
        tuple(value.get("dimensions", ())),
        taxonomy_comment,
    )


def _validate_reviewer_ledger(
    value: Any,
    *,
    packet: Mapping[str, Any],
    packet_sha256: str,
    protocol_sha256: str,
    codebook_sha256: str,
    dimensions: frozenset[str],
    expected_assignment: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, dict[str, Any]]]:
    required = {
        "artifact_kind",
        "role",
        "reviewer_slot",
        "assignment_id",
        "review_phase",
        "packet_id",
        "protocol_sha256",
        "packet_sha256",
        "codebook_sha256",
        "workbook_content_sha256",
        "reviewer_pseudonym",
        "human_attestation",
        "started_at",
        "completed_at",
        "decisions",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != required
        or value.get("artifact_kind") != LEDGER_KIND
        or value.get("role") != "reviewer"
    ):
        _fail("reviewer ledger schema, artifact kind, or role differs")
    bindings = {
        "review_phase": packet["review_phase"],
        "packet_id": packet["packet_id"],
        "protocol_sha256": protocol_sha256,
        "packet_sha256": packet_sha256,
        "codebook_sha256": codebook_sha256,
    }
    for field, expected in bindings.items():
        if value.get(field) != expected:
            _fail(f"reviewer ledger {field} binding differs")
    reviewer_slot = value.get("reviewer_slot")
    assignment_id = value.get("assignment_id")
    if reviewer_slot not in {"A", "B"}:
        _fail("reviewer ledger slot must be A or B")
    if (
        not isinstance(assignment_id, str)
        or re.fullmatch(r"assignment_[0-9a-f]{64}", assignment_id) is None
    ):
        _fail("reviewer ledger assignment ID is invalid")
    workbook_digest = value.get("workbook_content_sha256")
    if not isinstance(workbook_digest, str) or HEX64.fullmatch(workbook_digest) is None:
        _fail("reviewer ledger workbook content binding is invalid")
    if expected_assignment is not None:
        if (
            reviewer_slot != expected_assignment.get("reviewer_slot")
            or assignment_id != expected_assignment.get("assignment_id")
            or workbook_digest != expected_assignment.get("workbook_content_sha256")
        ):
            _fail("reviewer ledger belongs to another reviewer assignment")
    pseudonym = value.get("reviewer_pseudonym")
    if not isinstance(pseudonym, str) or PSEUDONYM.fullmatch(pseudonym) is None:
        _fail("reviewer ledger pseudonym is invalid")
    if value.get("human_attestation") != HUMAN_ATTESTATION:
        _fail("reviewer ledger lacks the required human-only attestation")
    started = _timestamp(value.get("started_at"), "reviewer ledger started_at")
    completed = _timestamp(value.get("completed_at"), "reviewer ledger completed_at")
    if datetime.fromisoformat(completed.replace("Z", "+00:00")) < datetime.fromisoformat(
        started.replace("Z", "+00:00")
    ):
        _fail("reviewer ledger completion precedes its start")
    raw_decisions = value.get("decisions")
    if not isinstance(raw_decisions, list):
        _fail("reviewer ledger decisions must be an array")
    case_ids = {case["case_id"] for case in packet["cases"]}
    if [row.get("case_id") if isinstance(row, Mapping) else None for row in raw_decisions] != [
        case["case_id"] for case in packet["cases"]
    ]:
        _fail("reviewer ledger decisions are not in exact packet case order")
    decisions: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_decisions):
        parsed = _decision(
            raw,
            context=f"reviewer ledger decisions[{index}]",
            case_ids=case_ids,
            dimensions=dimensions,
        )
        if parsed["case_id"] in decisions:
            _fail("reviewer ledger repeats a case decision")
        decisions[parsed["case_id"]] = parsed
    if set(decisions) != case_ids:
        _fail("reviewer ledger decisions do not cover every packet case exactly once")
    return pseudonym, decisions


def _workbook_content(
    *,
    packet_paths: Sequence[Path],
    protocol_path: Path,
    codebook_path: Path,
    reviewer_slot: str,
    secret: bytes,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    secret = _key(secret)
    if reviewer_slot not in {"A", "B"}:
        _fail("reviewer_slot must be A or B")
    _protocol, protocol_descriptor = _validate_protocol(protocol_path)
    codebook, codebook_descriptor, _dimensions = _validate_codebook(
        codebook_path, protocol_sha256=protocol_descriptor["sha256"]
    )
    if not packet_paths:
        _fail("review workbook requires at least one packet")
    packet_entries: list[dict[str, Any]] = []
    packets: list[dict[str, Any]] = []
    phases: set[str] = set()
    seen_packets: set[str] = set()
    for index, path in enumerate(packet_paths):
        path = Path(path).resolve()
        payload = _read(path)
        packet = _json_payload(payload, context=f"review packet {index}")
        packet_id, _cases = _validate_packet(
            packet,
            protocol_sha256=protocol_descriptor["sha256"],
            codebook_sha256=codebook_descriptor["sha256"],
            secret=secret,
        )
        if packet_id in seen_packets:
            _fail("review workbook repeats a packet")
        seen_packets.add(packet_id)
        phases.add(packet["review_phase"])
        packets.append(packet)
        packet_entries.append(
            {
                "packet_id": packet_id,
                "packet_sha256": _sha256(payload),
                "review_phase": packet["review_phase"],
                "case_count": len(packet["cases"]),
                "case_ids": [case["case_id"] for case in packet["cases"]],
            }
        )
    if len(phases) != 1:
        _fail("one workbook cannot mix pilot and full-review packets")
    phase = next(iter(phases))
    if phase == "full" and codebook["status"] != "frozen":
        _fail("full-review workbook export requires a frozen codebook")
    if phase == "pilot" and codebook["status"] != "pilot":
        _fail("pilot workbook requires the pilot codebook")

    task_by_digest: dict[str, dict[str, Any]] = {}
    for packet_index, packet in enumerate(packets):
        for case in packet["cases"]:
            exact_task = {"material": case["material"]}
            digest = _sha256(_canonical_bytes(exact_task))
            task_id = f"task_{digest}"
            if task_id not in task_by_digest:
                task_by_digest[task_id] = {
                    "task_id": task_id,
                    **exact_task,
                    "aliases": [],
                }
            task = task_by_digest[task_id]
            if task["material"] != case["material"]:
                _fail("workbook task digest collision")
            task["aliases"].append(
                {"packet_index": packet_index, "case_id": case["case_id"]}
            )
    assignment_basis = {
        "artifact_kind": WORKBOOK_KIND,
        "review_phase": phase,
        "protocol_sha256": protocol_descriptor["sha256"],
        "codebook_sha256": codebook_descriptor["sha256"],
        "packets": packet_entries,
        "task_ids": sorted(task_by_digest),
    }
    assignment_id = _opaque(
        secret,
        "assignment",
        {"reviewer_slot": reviewer_slot, **assignment_basis},
    )
    task_order = sorted(
        task_by_digest,
        key=lambda task_id: hmac.new(
            secret,
            f"{assignment_id}:{task_id}".encode("ascii"),
            hashlib.sha256,
        ).hexdigest(),
    )
    if reviewer_slot == "B" and len(task_order) > 1:
        a_assignment = _opaque(
            secret,
            "assignment",
            {"reviewer_slot": "A", **assignment_basis},
        )
        a_order = sorted(
            task_by_digest,
            key=lambda task_id: hmac.new(
                secret,
                f"{a_assignment}:{task_id}".encode("ascii"),
                hashlib.sha256,
            ).hexdigest(),
        )
        if task_order == a_order:
            task_order = task_order[1:] + task_order[:1]
    unsigned = {
        "artifact_kind": WORKBOOK_KIND,
        "reviewer_slot": reviewer_slot,
        "assignment_id": assignment_id,
        "review_phase": phase,
        "protocol_sha256": protocol_descriptor["sha256"],
        "codebook_sha256": codebook_descriptor["sha256"],
        "packets": packet_entries,
        "tasks": [task_by_digest[task_id] for task_id in task_order],
    }
    workbook_digest = _sha256(_canonical_bytes(unsigned))
    content = {**unsigned, "workbook_content_sha256": workbook_digest}
    support = {
        "codebook": _public_codebook(codebook),
        "human_attestation": HUMAN_ATTESTATION,
    }
    return content, packets, support


def _public_codebook(codebook: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the taxonomy/rubric fields a reviewer needs to see."""

    return {
        "artifact_kind": codebook["artifact_kind"],
        "status": codebook["status"],
        "protocol_sha256": codebook["protocol_sha256"],
        "classification_rubric": codebook["classification_rubric"],
        "dimensions": codebook["dimensions"],
        "frozen_at": codebook["frozen_at"],
    }


def _safe_script_json(value: Any) -> str:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _reviewer_html(content: Mapping[str, Any], support: Mapping[str, Any]) -> bytes:
    payload = _safe_script_json({"workbook": content, **support})
    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blinded INDRA claim-evidence review</title>
<style>
:root{font-family:ui-sans-serif,system-ui,sans-serif;color:#17202a;background:#f4f1ea}body{margin:0}main{max-width:1100px;margin:auto;padding:24px}.card{background:#fff;border:1px solid #d8d2c4;border-radius:12px;padding:20px;margin:16px 0;box-shadow:0 2px 12px #0001}.meta{color:#59636e;font-size:.92rem}.bar{height:10px;background:#e5e0d5;border-radius:9px;overflow:hidden}.bar>i{display:block;height:100%;background:#306b55}.claim,.evidence{white-space:pre-wrap;overflow-wrap:anywhere;background:#f8f7f3;border:1px solid #e5e0d5;border-radius:8px;padding:12px;margin:8px 0}.evidence{border-left:4px solid #6d7f91}.choice{display:block;margin:9px 0}.dimensions{columns:2;column-gap:24px}.dimensions label{display:block;break-inside:avoid;margin:7px 0}textarea,input[type=text]{width:100%;box-sizing:border-box;padding:10px;border:1px solid #a9b0b6;border-radius:6px}button{padding:10px 14px;border:0;border-radius:7px;background:#245b49;color:white;cursor:pointer;margin:4px}button.secondary{background:#56616b}button:disabled{opacity:.45;cursor:not-allowed}.warning{background:#fff4d6;border:1px solid #e2c66f;padding:12px;border-radius:8px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:700px){.dimensions{columns:1}.grid{grid-template-columns:1fr}}
</style></head><body><main>
<h1>Blinded claim-evidence review</h1>
<p class="warning">Human-only offline review. Use only the displayed claim and evidence. Do not use a model or external lookup. The workbook stores progress only in this browser and exports no data over a network.</p>
<section class="card" id="rubric"></section><section class="card" id="identity"></section>
<div class="bar"><i id="progress"></i></div><p class="meta" id="counter"></p>
<section class="card" id="case"></section><section class="card" id="decision"></section>
<div><button class="secondary" id="previous">Previous</button><button id="next">Save &amp; next</button><button class="secondary" id="checkpoint">Export checkpoint</button><label class="choice">Restore checkpoint <input id="restore" type="file" accept="application/json"></label><button id="export">Prepare complete packet ledgers</button></div><div id="downloads"></div>
</main><script id="payload" type="application/json">""" + payload + """</script>
<script>
'use strict';
const P=JSON.parse(document.getElementById('payload').textContent), W=P.workbook, C=P.codebook;
const storageKey='indra-error-review:'+W.assignment_id;
let state={index:0,pseudonym:'',attested:false,started_at:null,decisions:{}};
try{const saved=JSON.parse(localStorage.getItem(storageKey));if(saved&&typeof saved==='object')state={...state,...saved}}catch(_e){}
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pretty=x=>esc(JSON.stringify(x,null,2));
function save(){try{localStorage.setItem(storageKey,JSON.stringify(state))}catch(_e){document.querySelector('.warning').textContent+=' Browser storage is unavailable; export checkpoints frequently.'}}
function currentDecision(){if(!W.tasks.length)return null;const id=W.tasks[state.index].task_id;return state.decisions[id]||{classification:'',dimensions:[],comment:''}}
function capture(){if(!W.tasks.length)return;const task=W.tasks[state.index],dims=[...document.querySelectorAll('input[name=dimension]:checked')].map(x=>x.value);state.decisions[task.task_id]={classification:document.querySelector('input[name=classification]:checked')?.value||'',dimensions:dims.sort(),comment:document.getElementById('comment').value};if(!state.started_at)state.started_at=new Date().toISOString();save()}
function render(){if(!W.tasks.length){document.getElementById('counter').textContent='No threshold errors require review.';document.getElementById('progress').style.width='100%';document.getElementById('case').innerHTML='<h2>No cases</h2>';document.getElementById('decision').innerHTML='';document.getElementById('previous').disabled=true;document.getElementById('next').disabled=true;return}const t=W.tasks[state.index],d=currentDecision();document.getElementById('counter').textContent=`Task ${state.index+1} of ${W.tasks.length}; ${t.aliases.length} exact packet case(s).`;document.getElementById('progress').style.width=`${100*(state.index+1)/W.tasks.length}%`;document.getElementById('case').innerHTML=`<h2>Assembled claim</h2><div class=claim>${pretty(t.material.claim)}</div><h3>Exact panel evidence (${t.material.evidence.length})</h3>`+t.material.evidence.map((e,i)=>`<div class=evidence><b>Evidence ${i+1}</b><br>${pretty(e)}</div>`).join('');document.getElementById('decision').innerHTML='<h2>Human classification</h2>'+['supports_claim','rejects_claim','indeterminate'].map(k=>`<label class=choice><input type=radio name=classification value="${esc(k)}" ${d.classification===k?'checked':''}> <b>${esc(k)}</b> — ${esc(C.classification_rubric[k])}</label>`).join('')+'<h3>Dimensions (one or more)</h3><div class=dimensions>'+C.dimensions.map(x=>`<label title="${esc(x.definition)}"><input type=checkbox name=dimension value="${esc(x.dimension)}" ${d.dimensions.includes(x.dimension)?'checked':''}> <b>${esc(x.label)}</b><br><span class=meta>${esc(x.definition)}</span></label>`).join('')+'</div><h3>Optional taxonomy/refinement comment</h3><textarea id=comment maxlength=4000 rows=4>'+esc(d.comment)+'</textarea>';document.getElementById('previous').disabled=state.index===0;document.getElementById('next').textContent=state.index===W.tasks.length-1?'Save':'Save & next'}
function validDecision(d){const allowed=new Set(C.dimensions.map(x=>x.dimension));return d&&['supports_claim','rejects_claim','indeterminate'].includes(d.classification)&&Array.isArray(d.dimensions)&&d.dimensions.length>0&&new Set(d.dimensions).size===d.dimensions.length&&d.dimensions.every(x=>allowed.has(x))&&String(d.comment||'').length<=4000&&(!d.dimensions.includes('taxonomy_gap')||String(d.comment||'').trim())}
function download(name,value){const blob=new Blob([JSON.stringify(value,null,2)+'\\n'],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
document.getElementById('previous').onclick=()=>{capture();if(state.index>0)state.index--;save();render()};document.getElementById('next').onclick=()=>{capture();if(state.index<W.tasks.length-1)state.index++;save();render()};
document.getElementById('checkpoint').onclick=()=>{capture();download('checkpoint_'+W.assignment_id.slice(11,27)+'.json',{assignment_id:W.assignment_id,state})};document.getElementById('restore').onchange=async e=>{try{const v=JSON.parse(await e.target.files[0].text());if(v.assignment_id!==W.assignment_id||!v.state||typeof v.state!=='object')throw new Error();state={...state,...v.state};save();render()}catch(_e){alert('Checkpoint does not belong to this assignment.')}};
document.getElementById('export').onclick=()=>{capture();if(!/^[A-Za-z0-9][A-Za-z0-9._-]{2,63}$/.test(state.pseudonym)){alert('Enter a 3–64 character reviewer pseudonym.');return}if(state.attested!==true){alert('Affirm the human-only attestation before export.');return}if(W.review_phase==='full'&&C.status!=='frozen'){alert('Full-review export requires a frozen codebook.');return}for(const t of W.tasks)if(!validDecision(state.decisions[t.task_id])){alert('Every task needs a classification and at least one valid dimension.');return}const completed=new Date().toISOString(),ledgers=[];for(let p=0;p<W.packets.length;p++){const packet=W.packets[p],byCase={};for(const t of W.tasks){for(const a of t.aliases)if(a.packet_index===p){const d=state.decisions[t.task_id];byCase[a.case_id]={classification:d.classification,dimensions:[...d.dimensions].sort(),comment:String(d.comment||'').trim()||null}}}const decisions=packet.case_ids.map(case_id=>({case_id,...byCase[case_id]}));ledgers.push({artifact_kind:'indra_belief_error_review_ledger',role:'reviewer',reviewer_slot:W.reviewer_slot,assignment_id:W.assignment_id,review_phase:W.review_phase,packet_id:packet.packet_id,protocol_sha256:W.protocol_sha256,packet_sha256:packet.packet_sha256,codebook_sha256:W.codebook_sha256,workbook_content_sha256:W.workbook_content_sha256,reviewer_pseudonym:state.pseudonym,human_attestation:P.human_attestation,started_at:state.started_at||completed,completed_at:completed,decisions})}const area=document.getElementById('downloads');area.innerHTML='<p>Validated. Download each bound packet ledger:</p>';ledgers.forEach((ledger,i)=>{const b=document.createElement('button');b.textContent='Download ledger '+(i+1);b.onclick=()=>download('ledger_'+W.reviewer_slot+'_'+ledger.packet_id.slice(7,23)+'.json',ledger);area.appendChild(b)})};
document.getElementById('rubric').innerHTML='<h2>Classification rubric</h2><ol>'+C.classification_rubric.ordered_steps.map(x=>'<li>'+esc(x)+'</li>').join('')+'</ol><p class=meta>Codebook status: '+esc(C.status)+' · Phase: '+esc(W.review_phase)+' · Reviewer assignment: '+esc(W.reviewer_slot)+' · Exact byte-identical tasks are shown once and expanded into complete per-packet ledgers on export.</p>';
document.getElementById('identity').innerHTML='<h2>Reviewer identity</h2><label>Pseudonym (do not use a name or email)<input id=pseudonym type=text maxlength=64 value="'+esc(state.pseudonym)+'"></label><label class=choice><input id=attestation type=checkbox '+(state.attested?'checked':'')+'> '+esc(P.human_attestation)+'</label>';document.getElementById('pseudonym').oninput=e=>{state.pseudonym=e.target.value;save()};document.getElementById('attestation').onchange=e=>{state.attested=e.target.checked;save()};render();
</script></body></html>"""
    return document.encode("utf-8")


def generate_reviewer_workbook(
    *,
    packet_paths: Sequence[Path],
    protocol_path: Path,
    codebook_path: Path,
    blinding_key: bytes,
    output_dir: Path,
) -> dict[str, Any]:
    """Publish independent A/B self-contained workbooks for one or more packets."""

    secret = _key(blinding_key)
    _protocol, protocol_descriptor = _validate_protocol(protocol_path)
    codebook, _codebook_descriptor, _dimensions = _validate_codebook(
        codebook_path, protocol_sha256=protocol_descriptor["sha256"]
    )
    if codebook["status"] == "frozen":
        _validate_frozen_codebook_authenticity(codebook, secret=secret)
    output_dir = Path(output_dir).resolve()
    workbooks: list[dict[str, Any]] = []
    for reviewer_slot in ("A", "B"):
        content, _packets, support = _workbook_content(
            packet_paths=packet_paths,
            protocol_path=protocol_path,
            codebook_path=codebook_path,
            reviewer_slot=reviewer_slot,
            secret=secret,
        )
        payload = _reviewer_html(content, support)
        path = output_dir / (
            f"workbook_{reviewer_slot}_{content['workbook_content_sha256']}.html"
        )
        _atomic_write(path, payload, private=False)
        workbooks.append(
            {
                "reviewer_slot": reviewer_slot,
                "assignment_id": content["assignment_id"],
                "workbook": path,
                "workbook_sha256": _sha256(payload),
                "workbook_content_sha256": content["workbook_content_sha256"],
                "unique_task_count": len(content["tasks"]),
            }
        )
    return {
        "workbooks": workbooks,
        "packet_count": len(content["packets"]),
        "expanded_case_count": sum(row["case_count"] for row in content["packets"]),
    }


def _validate_reviewer_workbooks(
    *,
    packet_paths: Sequence[Path],
    reviewer_workbook_paths: Sequence[Path],
    target_packet: Mapping[str, Any],
    target_packet_sha256: str,
    protocol_path: Path,
    codebook_path: Path,
    secret: bytes,
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    secret = _key(secret)
    if len(reviewer_workbook_paths) != 2:
        _fail("exactly two reviewer workbooks are required in A, B order")
    assignments: dict[str, dict[str, Any]] = {}
    workbook_descriptors: list[dict[str, Any]] = []
    packet_commitments: list[dict[str, Any]] | None = None
    for reviewer_slot, supplied_path in zip(
        ("A", "B"), reviewer_workbook_paths, strict=True
    ):
        content, _packets, support = _workbook_content(
            packet_paths=packet_paths,
            protocol_path=protocol_path,
            codebook_path=codebook_path,
            reviewer_slot=reviewer_slot,
            secret=secret,
        )
        matches = [
            row for row in content["packets"]
            if row["packet_id"] == target_packet["packet_id"]
        ]
        if len(matches) != 1 or matches[0]["packet_sha256"] != target_packet_sha256:
            _fail("reviewer workbook packet list lacks the exact target packet")
        expected_payload = _reviewer_html(content, support)
        supplied_path = Path(supplied_path).resolve()
        supplied_payload = _read(supplied_path)
        if supplied_payload != expected_payload:
            _fail(f"reviewer workbook {reviewer_slot} is not the exact canonical HTML")
        if packet_commitments is None:
            packet_commitments = content["packets"]
        elif packet_commitments != content["packets"]:
            _fail("reviewer assignment packet commitments differ")
        assignments[reviewer_slot] = {
            "reviewer_slot": reviewer_slot,
            "assignment_id": content["assignment_id"],
            "workbook_content_sha256": content["workbook_content_sha256"],
        }
        workbook_descriptors.append(_descriptor(supplied_path, supplied_payload))
    packet_descriptors = [_descriptor(Path(path).resolve()) for path in packet_paths]
    return assignments, workbook_descriptors, packet_descriptors, packet_commitments or []


def freeze_codebook(
    *,
    protocol_path: Path,
    pilot_codebook_path: Path,
    candidate_codebook_path: Path,
    pilot_packet_path: Path,
    pilot_admin_manifest_path: Path,
    pilot_workbook_paths: Sequence[Path],
    reviewer_ledger_paths: Sequence[Path],
    blinding_key: bytes,
    human_freeze_attested: bool,
    frozen_at: str,
    output_path: Path,
) -> dict[str, Any]:
    """Freeze a human-refined taxonomy only after two complete pilot ledgers."""

    if human_freeze_attested is not True:
        _fail("codebook freeze requires an explicit human freeze attestation")
    secret = _key(blinding_key)
    _protocol, protocol_descriptor = _validate_protocol(protocol_path)
    pilot_codebook, pilot_descriptor, pilot_dimensions = _validate_codebook(
        pilot_codebook_path, protocol_sha256=protocol_descriptor["sha256"]
    )
    candidate, candidate_descriptor, _candidate_dimensions = _validate_codebook(
        candidate_codebook_path, protocol_sha256=protocol_descriptor["sha256"]
    )
    if pilot_codebook["status"] != "pilot" or candidate["status"] != "pilot":
        _fail("codebook freeze requires pilot-status source and candidate codebooks")
    if (
        pilot_codebook["classification_rubric"]
        != {"ordered_steps": CLASSIFICATION_STEPS, **CLASSIFICATION_DEFINITIONS}
        or pilot_codebook["dimensions"] != _default_dimensions()
        or pilot_codebook["pilot_contract"] != PILOT_CONTRACT
    ):
        _fail("source pilot codebook is not the canonical pre-pilot codebook")
    if candidate["classification_rubric"] != pilot_codebook["classification_rubric"]:
        _fail("the human pilot may refine dimensions but not the frozen classification semantics")
    if candidate["pilot_contract"] != pilot_codebook["pilot_contract"]:
        _fail("the human-refined candidate cannot change the pilot contract")
    packet_path = Path(pilot_packet_path).resolve()
    packet_payload = _read(packet_path)
    packet = _json_payload(packet_payload, context="pilot review packet")
    _packet_id, cases = _validate_packet(
        packet,
        protocol_sha256=protocol_descriptor["sha256"],
        codebook_sha256=pilot_descriptor["sha256"],
        secret=secret,
    )
    if packet["review_phase"] != "pilot" or not cases:
        _fail("codebook freeze requires a non-empty pilot packet")
    admin_path = Path(pilot_admin_manifest_path).resolve()
    admin_payload = _read(admin_path)
    admin = _json_payload(admin_payload, context="pilot admin manifest")
    _validate_admin(
        admin,
        packet_path=packet_path,
        packet=packet,
        packet_payload=packet_payload,
        protocol_descriptor=protocol_descriptor,
        codebook_descriptor=pilot_descriptor,
        secret=secret,
    )
    assignments, workbook_descriptors, _packet_descriptors, _packet_commitments = (
        _validate_reviewer_workbooks(
            packet_paths=[packet_path],
            reviewer_workbook_paths=pilot_workbook_paths,
            target_packet=packet,
            target_packet_sha256=_sha256(packet_payload),
            protocol_path=protocol_path,
            codebook_path=pilot_codebook_path,
            secret=secret,
        )
    )
    if len(reviewer_ledger_paths) != 2:
        _fail("codebook freeze requires exactly two independent pilot ledgers")
    reviewers: list[str] = []
    ledger_descriptors: list[dict[str, Any]] = []
    completed_times: list[str] = []
    slots: set[str] = set()
    for index, ledger_path in enumerate(reviewer_ledger_paths):
        ledger_path = Path(ledger_path).resolve()
        ledger_payload = _read(ledger_path)
        ledger = _json_payload(ledger_payload, context=f"pilot reviewer ledger {index}")
        pseudonym, _decisions = _validate_reviewer_ledger(
            ledger,
            packet=packet,
            packet_sha256=_sha256(packet_payload),
            protocol_sha256=protocol_descriptor["sha256"],
            codebook_sha256=pilot_descriptor["sha256"],
            dimensions=pilot_dimensions,
            expected_assignment=assignments.get(ledger.get("reviewer_slot")),
        )
        reviewers.append(pseudonym)
        slots.add(ledger["reviewer_slot"])
        completed_times.append(ledger["completed_at"])
        ledger_descriptors.append(_descriptor(ledger_path, ledger_payload))
    if reviewers[0].casefold() == reviewers[1].casefold():
        _fail("pilot reviewers must use distinct pseudonyms")
    if slots != {"A", "B"}:
        _fail("pilot reviewer ledgers must cover assignments A and B exactly once")

    freeze_timestamp = _timestamp(frozen_at, "frozen_at")
    freeze_moment = datetime.fromisoformat(freeze_timestamp.replace("Z", "+00:00"))
    prerequisite_moments = [
        datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        for timestamp in [admin["created_at"], *completed_times]
    ]
    if freeze_moment < max(prerequisite_moments):
        _fail("frozen_at must not precede pilot preparation or either completed review")

    value = dict(candidate)
    value["status"] = "frozen"
    value["frozen_at"] = freeze_timestamp
    value["pilot_provenance"] = {
        "protocol": protocol_descriptor,
        "source_pilot_codebook": pilot_descriptor,
        "human_refined_candidate": candidate_descriptor,
        "pilot_packet": _descriptor(packet_path, packet_payload),
        "pilot_admin_manifest": _descriptor(admin_path, admin_payload),
        "reviewer_ledgers": ledger_descriptors,
        "reviewer_workbooks": workbook_descriptors,
    }
    value["human_freeze_attestation"] = FREEZE_ATTESTATION
    unsigned = dict(value)
    unsigned.pop("freeze_binding")
    value["freeze_binding"] = _opaque(secret, "freeze", unsigned)
    write_json(value, output_path)
    payload = _read(output_path)
    return {
        "codebook": Path(output_path).resolve(),
        "codebook_sha256": _sha256(payload),
        "dimension_count": len(value["dimensions"]),
        "pilot_case_count": len(cases),
    }


def _validate_admin(
    value: Any,
    *,
    packet_path: Path,
    packet: Mapping[str, Any],
    packet_payload: bytes,
    protocol_descriptor: Mapping[str, Any],
    codebook_descriptor: Mapping[str, Any],
    secret: bytes,
) -> dict[str, Any]:
    required = {
        "artifact_kind",
        "created_at",
        "input_binding",
        "scope_id",
        "provenance",
        "protocol",
        "codebook",
        "packet",
        "packet_id",
        "threshold_error_census",
        "threshold_error_statement_ids",
        "case_mapping",
        "admin_id",
    }
    if not isinstance(value, Mapping) or set(value) != required or value.get("artifact_kind") != ADMIN_KIND:
        _fail("admin manifest schema or artifact kind differs")
    _timestamp(value.get("created_at"), "admin manifest created_at")
    unsigned = dict(value)
    admin_id = unsigned.pop("admin_id")
    if (
        not isinstance(admin_id, str)
        or OPAQUE_ID.fullmatch(admin_id) is None
        or not admin_id.startswith("admin_")
        or admin_id != _opaque(secret, "admin", unsigned)
    ):
        _fail("admin manifest binding is invalid")
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        _fail("admin manifest provenance must be an object")
    if value.get("input_binding") != _opaque(secret, "input", provenance):
        _fail("admin manifest input binding differs")
    if value.get("scope_id") != _opaque(secret, "scope", value["input_binding"]):
        _fail("admin manifest scope binding differs")
    if value.get("input_binding") != packet.get("input_binding") or value.get("scope_id") != packet.get("scope_id"):
        _fail("admin manifest belongs to another review packet")
    if value.get("packet_id") != packet.get("packet_id"):
        _fail("admin manifest packet ID differs")
    if value.get("protocol") != protocol_descriptor or value.get("codebook") != codebook_descriptor:
        _fail("admin manifest protocol or codebook descriptor differs")
    described_packet_path, described_packet_payload = _check_descriptor(
        value.get("packet"), context="admin packet"
    )
    if described_packet_path != Path(packet_path).resolve() or described_packet_payload != packet_payload:
        _fail("admin manifest packet descriptor differs")
    _validate_packet(
        packet,
        protocol_sha256=protocol_descriptor["sha256"],
        codebook_sha256=codebook_descriptor["sha256"],
        secret=secret,
    )

    files = provenance.get("files")
    if not isinstance(files, Mapping) or set(files) != {
        "aggregation_config", "bundle_manifest", "execution_ledger",
        "execution_map", "gold", "predictions", "pricing_config", "protocol",
        "raw_attempts", "spec", "spend_ledger", "statements",
    }:
        _fail("admin provenance file set differs")
    spec_path = Path(files["spec"]["path"])
    bundle_path = Path(files["bundle_manifest"]["path"])
    protocol_path = Path(files["protocol"]["path"])
    rebuilt = _bundle_scope(
        spec_path=spec_path,
        bundle_manifest_path=bundle_path,
        panel_id=provenance.get("panel_id"),
        arm_id=provenance.get("arm_id"),
        protocol_path=protocol_path,
    )
    if rebuilt["provenance"] != provenance:
        _fail("admin provenance does not reproduce from its bound inputs")

    mapping = value.get("case_mapping")
    threshold_ids = value.get("threshold_error_statement_ids")
    if not isinstance(mapping, list) or not isinstance(threshold_ids, list):
        _fail("admin identity mappings must be arrays")
    packet_by_case = {case["case_id"]: case for case in packet["cases"]}
    if [row.get("case_id") if isinstance(row, Mapping) else None for row in mapping] != [
        case["case_id"] for case in packet["cases"]
    ]:
        _fail("admin case mapping is not in exact packet case order")
    mapped: set[str] = set()
    for index, row in enumerate(mapping):
        if not isinstance(row, Mapping) or set(row) != {
            "case_id", "statement_id", "reference_label", "probability_correct",
            "error_type", "execution_identities",
        }:
            _fail(f"admin case mapping {index} schema differs")
        case_id = row.get("case_id")
        statement_id = _identifier(row.get("statement_id"), f"admin case mapping {index}.statement_id")
        if (
            case_id not in packet_by_case
            or case_id in mapped
            or case_id != _opaque(secret, "case", [value["input_binding"], statement_id])
        ):
            _fail(f"admin case mapping {index} binding differs")
        _label(row.get("reference_label"), f"admin case mapping {index}.reference_label")
        _probability(row.get("probability_correct"), f"admin case mapping {index}.probability_correct")
        execution_ids = row.get("execution_identities")
        if not isinstance(execution_ids, list) or not execution_ids or any(
            not isinstance(item, str) or HEX64.fullmatch(item) is None for item in execution_ids
        ):
            _fail(f"admin case mapping {index} execution identities differ")
        if (
            rebuilt["gold"].get(statement_id) != row["reference_label"]
            or rebuilt["predictions"].get(statement_id) != row["probability_correct"]
            or rebuilt["execution_ids_by_statement"].get(statement_id) != execution_ids
            or row["error_type"]
            != (
                "false_positive"
                if int(row["probability_correct"] >= provenance["threshold"]) == 1
                else "false_negative"
            )
        ):
            _fail(f"admin case mapping {index} differs from the bound comparison inputs")
        mapped.add(case_id)
    if mapped != set(packet_by_case):
        _fail("admin mapping does not cover every packet case exactly once")
    expected_threshold_ids = [
        statement_id
        for statement_id in rebuilt["gold_order"]
        if int(rebuilt["predictions"][statement_id] >= provenance["threshold"])
        != rebuilt["gold"][statement_id]
    ]
    census = value.get("threshold_error_census")
    expected_census = {
        "count": len(expected_threshold_ids),
        "false_positive": sum(
            1
            for statement_id in expected_threshold_ids
            if rebuilt["predictions"][statement_id] >= provenance["threshold"]
        ),
        "false_negative": sum(
            1
            for statement_id in expected_threshold_ids
            if rebuilt["predictions"][statement_id] < provenance["threshold"]
        ),
    }
    if census != expected_census or packet["threshold_error_count"] != expected_census["count"]:
        _fail("admin threshold-error census differs from the bound comparison inputs")
    if (
        len(threshold_ids) != packet["threshold_error_count"]
        or len(set(threshold_ids)) != len(threshold_ids)
        or any(not isinstance(item, str) or UUID.fullmatch(item) is None for item in threshold_ids)
        or threshold_ids != expected_threshold_ids
    ):
        _fail("admin threshold-error census identity mapping differs")
    expected_errors: list[dict[str, Any]] = []
    for statement_id in expected_threshold_ids:
        probability = rebuilt["predictions"][statement_id]
        direction = (
            "false_positive"
            if probability >= provenance["threshold"]
            else "false_negative"
        )
        expected_errors.append(
            {
                "case_id": _opaque(
                    secret, "case", [value["input_binding"], statement_id]
                ),
                "material": rebuilt["material"][statement_id],
                "_error_type": direction,
            }
        )
    expected_selected = (
        _select_pilot_cases(expected_errors, count=packet["review_case_count"])
        if packet["review_phase"] == "pilot"
        else expected_errors
    )
    expected_public = [
        {
            "case_id": row["case_id"],
            "material": row["material"],
        }
        for row in expected_selected
    ]
    if packet["cases"] != expected_public:
        _fail("review packet cases are not the exact authenticated error selection")
    return rebuilt


def _validate_frozen_codebook_authenticity(
    codebook: Mapping[str, Any], *, secret: bytes
) -> None:
    """Authenticate the human pilot chain behind a frozen codebook."""

    if codebook.get("status") != "frozen":
        _fail("frozen codebook authentication requires a frozen codebook")
    unsigned_codebook = dict(codebook)
    freeze_binding = unsigned_codebook.pop("freeze_binding", None)
    if freeze_binding != _opaque(secret, "freeze", unsigned_codebook):
        _fail("frozen codebook authentication binding differs")
    provenance = codebook.get("pilot_provenance")
    if not isinstance(provenance, Mapping):
        _fail("frozen codebook lacks pilot provenance")
    protocol_path, protocol_payload = _check_descriptor(
        provenance.get("protocol"), context="frozen pilot protocol"
    )
    _protocol, protocol_descriptor = _validate_protocol(protocol_path)
    if protocol_descriptor["sha256"] != _sha256(protocol_payload):
        _fail("frozen pilot protocol descriptor differs")
    pilot_codebook_path, _pilot_codebook_payload = _check_descriptor(
        provenance.get("source_pilot_codebook"), context="frozen pilot codebook"
    )
    _pilot, pilot_descriptor, dimensions = _validate_codebook(
        pilot_codebook_path, protocol_sha256=protocol_descriptor["sha256"]
    )
    packet_path, packet_payload = _check_descriptor(
        provenance.get("pilot_packet"), context="frozen pilot packet"
    )
    packet = _json_payload(packet_payload, context="frozen pilot packet")
    _validate_packet(
        packet,
        protocol_sha256=protocol_descriptor["sha256"],
        codebook_sha256=pilot_descriptor["sha256"],
        secret=secret,
    )
    _admin_path, admin_payload = _check_descriptor(
        provenance.get("pilot_admin_manifest"), context="frozen pilot admin manifest"
    )
    admin = _json_payload(admin_payload, context="frozen pilot admin manifest")
    _validate_admin(
        admin,
        packet_path=packet_path,
        packet=packet,
        packet_payload=packet_payload,
        protocol_descriptor=protocol_descriptor,
        codebook_descriptor=pilot_descriptor,
        secret=secret,
    )
    workbook_descriptors = provenance.get("reviewer_workbooks")
    ledger_descriptors = provenance.get("reviewer_ledgers")
    if (
        not isinstance(workbook_descriptors, list)
        or len(workbook_descriptors) != 2
        or not isinstance(ledger_descriptors, list)
        or len(ledger_descriptors) != 2
    ):
        _fail("frozen pilot reviewer provenance differs")
    workbook_paths = [
        _check_descriptor(
            descriptor, context=f"frozen pilot reviewer workbook {index}"
        )[0]
        for index, descriptor in enumerate(workbook_descriptors)
    ]
    assignments, _workbooks, _packet_descriptors, _packet_commitments = (
        _validate_reviewer_workbooks(
            packet_paths=[packet_path],
            reviewer_workbook_paths=workbook_paths,
            target_packet=packet,
            target_packet_sha256=_sha256(packet_payload),
            protocol_path=protocol_path,
            codebook_path=pilot_codebook_path,
            secret=secret,
        )
    )
    reviewers: list[str] = []
    slots: set[str] = set()
    for index, descriptor in enumerate(ledger_descriptors):
        _ledger_path, ledger_payload = _check_descriptor(
            descriptor, context=f"frozen pilot reviewer ledger {index}"
        )
        ledger = _json_payload(
            ledger_payload, context=f"frozen pilot reviewer ledger {index}"
        )
        slot = ledger.get("reviewer_slot")
        if slot not in assignments or slot in slots:
            _fail("frozen pilot ledgers must cover assignments A and B")
        reviewer, _decisions = _validate_reviewer_ledger(
            ledger,
            packet=packet,
            packet_sha256=_sha256(packet_payload),
            protocol_sha256=protocol_descriptor["sha256"],
            codebook_sha256=pilot_descriptor["sha256"],
            dimensions=dimensions,
            expected_assignment=assignments[slot],
        )
        reviewers.append(reviewer)
        slots.add(slot)
    if slots != {"A", "B"} or reviewers[0].casefold() == reviewers[1].casefold():
        _fail("frozen pilot reviewers are not independent")


def _resolver_html(
    workload: Mapping[str, Any], *, workload_sha256: str, codebook: Mapping[str, Any]
) -> bytes:
    payload = _safe_script_json(
        {
            "workload": workload,
            "workload_sha256": workload_sha256,
            "codebook": codebook,
            "human_attestation": HUMAN_ATTESTATION,
        }
    )
    document = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Blinded INDRA disagreement resolution</title>
<style>:root{font-family:system-ui,sans-serif;background:#f4f1ea;color:#17202a}main{max-width:1050px;margin:auto;padding:24px}.card{background:white;border:1px solid #d8d2c4;border-radius:10px;padding:18px;margin:14px 0}.material,pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f8f7f3;padding:10px;border-radius:7px}label{display:block;margin:8px 0}.dims{columns:2}textarea,input[type=text]{width:100%;box-sizing:border-box;padding:9px}button{padding:10px 14px;background:#245b49;color:white;border:0;border-radius:7px;margin:5px}.warning{background:#fff4d6;padding:12px;border:1px solid #e2c66f}</style></head><body><main><h1>Outcome-blind disagreement resolution</h1><p class=warning>This workload contains every reviewer disagreement and no agreement. Use only the displayed claim, evidence, independent classifications, and frozen rubric; do not use a model or external lookup.</p><section id=identity class=card></section><p id=count></p><section id=case class=card></section><section id=decision class=card></section><button id=prev>Previous</button><button id=next>Save &amp; next</button><button id=checkpoint>Export checkpoint</button><label>Restore checkpoint <input id=restore type=file accept="application/json"></label><button id=export>Export complete resolver ledger</button></main><script id=payload type="application/json">""" + payload + """</script><script>
'use strict';
const P=JSON.parse(document.getElementById('payload').textContent),W=P.workload,C=P.codebook,key='indra-error-resolver:'+W.workload_id;
let S={index:0,pseudonym:'',attested:false,started_at:null,decisions:{}};
try{const saved=JSON.parse(localStorage.getItem(key));if(saved&&typeof saved==='object')S={...S,...saved}}catch(_e){}
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pretty=x=>esc(JSON.stringify(x,null,2));
function save(){try{localStorage.setItem(key,JSON.stringify(S))}catch(_e){document.querySelector('.warning').textContent+=' Browser storage is unavailable; export checkpoints frequently.'}}
function capture(){const c=W.cases[S.index];S.decisions[c.case_id]={classification:document.querySelector('input[name=classification]:checked')?.value||'',dimensions:[...document.querySelectorAll('input[name=dimension]:checked')].map(x=>x.value).sort(),comment:document.getElementById('comment').value};if(!S.started_at)S.started_at=new Date().toISOString();save()}
function render(){const c=W.cases[S.index],d=S.decisions[c.case_id]||{classification:'',dimensions:[],comment:''};document.getElementById('count').textContent='Disagreement '+(S.index+1)+' of '+W.cases.length;document.getElementById('case').innerHTML='<h2>Assembled claim</h2><pre>'+pretty(c.material.claim)+'</pre><h3>Evidence</h3>'+c.material.evidence.map((x,i)=>'<div class=material><b>Evidence '+(i+1)+'</b><pre>'+pretty(x)+'</pre></div>').join('')+'<h3>Independent classifications</h3>'+c.reviewer_decisions.map((x,i)=>'<div class=material><b>Reviewer '+(i?'B':'A')+':</b> '+esc(x.classification)+' · '+esc(x.dimensions.join(', '))+(x.comment?'<br>Comment: '+esc(x.comment):'')+'</div>').join('');document.getElementById('decision').innerHTML='<h2>Final classification</h2>'+['supports_claim','rejects_claim','indeterminate'].map(k=>'<label><input type=radio name=classification value="'+esc(k)+'" '+(d.classification===k?'checked':'')+'> <b>'+esc(k)+'</b> — '+esc(C.classification_rubric[k])+'</label>').join('')+'<h3>Final dimensions</h3><div class=dims>'+C.dimensions.map(x=>'<label title="'+esc(x.definition)+'"><input type=checkbox name=dimension value="'+esc(x.dimension)+'" '+(d.dimensions.includes(x.dimension)?'checked':'')+'> '+esc(x.label)+'</label>').join('')+'</div><h3>Optional taxonomy/refinement comment</h3><textarea id=comment maxlength=4000 rows=4>'+esc(d.comment)+'</textarea>';document.getElementById('prev').disabled=S.index===0}
function valid(d){const allowed=new Set(C.dimensions.map(x=>x.dimension));return d&&['supports_claim','rejects_claim','indeterminate'].includes(d.classification)&&Array.isArray(d.dimensions)&&d.dimensions.length>0&&new Set(d.dimensions).size===d.dimensions.length&&d.dimensions.every(x=>allowed.has(x))&&String(d.comment||'').length<=4000&&(!d.dimensions.includes('taxonomy_gap')||String(d.comment||'').trim())}
function download(name,v){const a=document.createElement('a'),b=new Blob([JSON.stringify(v,null,2)+'\\n'],{type:'application/json'});a.href=URL.createObjectURL(b);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
document.getElementById('prev').onclick=()=>{capture();if(S.index)S.index--;save();render()};
document.getElementById('next').onclick=()=>{capture();if(S.index<W.cases.length-1)S.index++;save();render()};
document.getElementById('checkpoint').onclick=()=>{capture();download('resolver_checkpoint_'+W.workload_id.slice(9,25)+'.json',{workload_id:W.workload_id,state:S})};
document.getElementById('restore').onchange=async e=>{try{const v=JSON.parse(await e.target.files[0].text());if(v.workload_id!==W.workload_id||!v.state)throw new Error();S={...S,...v.state};save();render()}catch(_e){alert('Checkpoint does not belong to this workload.')}};
document.getElementById('export').onclick=()=>{capture();if(!/^[A-Za-z0-9][A-Za-z0-9._-]{2,63}$/.test(S.pseudonym)){alert('Enter a 3–64 character resolver pseudonym.');return}if(S.attested!==true){alert('Affirm the human-only attestation before export.');return}if(!W.cases.every(c=>valid(S.decisions[c.case_id]))){alert('Resolve every disagreement with a classification and dimension.');return}const completed=new Date().toISOString(),decisions=W.cases.map(c=>{const d=S.decisions[c.case_id];return{case_id:c.case_id,classification:d.classification,dimensions:[...d.dimensions].sort(),comment:String(d.comment||'').trim()||null}});download('resolver_ledger_'+W.workload_id.slice(9,25)+'.json',{artifact_kind:'indra_belief_error_review_ledger',role:'resolver',review_phase:W.review_phase,packet_id:W.packet_id,protocol_sha256:W.protocol_sha256,packet_sha256:W.packet_sha256,codebook_sha256:W.codebook_sha256,resolver_workload_sha256:P.workload_sha256,reviewer_ledger_sha256s:W.reviewer_ledger_sha256s,resolver_pseudonym:S.pseudonym,human_attestation:P.human_attestation,started_at:S.started_at||completed,completed_at:completed,decisions})};
document.getElementById('identity').innerHTML='<label>Resolver pseudonym (do not use a name or email)<input id=pseudonym type=text maxlength=64 value="'+esc(S.pseudonym)+'"></label><label><input id=attestation type=checkbox '+(S.attested?'checked':'')+'> '+esc(P.human_attestation)+'</label>';document.getElementById('pseudonym').oninput=e=>{S.pseudonym=e.target.value;save()};document.getElementById('attestation').onchange=e=>{S.attested=e.target.checked;save()};render();
</script></body></html>"""
    return document.encode("utf-8")

def generate_resolver_workload(
    *,
    packet_path: Path,
    protocol_path: Path,
    codebook_path: Path,
    reviewer_ledger_paths: Sequence[Path],
    reviewer_workbook_packet_paths: Sequence[Path],
    reviewer_workbook_paths: Sequence[Path],
    blinding_key: bytes,
    output_dir: Path,
) -> dict[str, Any]:
    """Publish an outcome-blind workload containing every disagreement and no agreement."""

    secret = _key(blinding_key)
    _protocol, protocol_descriptor = _validate_protocol(protocol_path)
    codebook, codebook_descriptor, dimensions = _validate_codebook(
        codebook_path, protocol_sha256=protocol_descriptor["sha256"]
    )
    if codebook["status"] != "frozen":
        _fail("resolver workload requires a frozen codebook")
    _validate_frozen_codebook_authenticity(codebook, secret=secret)
    packet_path = Path(packet_path).resolve()
    packet_payload = _read(packet_path)
    packet = _json_payload(packet_payload, context="full review packet")
    _packet_id, cases = _validate_packet(
        packet,
        protocol_sha256=protocol_descriptor["sha256"],
        codebook_sha256=codebook_descriptor["sha256"],
        secret=secret,
    )
    if packet["review_phase"] != "full":
        _fail("resolver workload requires a full-review packet")
    (
        assignments,
        workbook_descriptors,
        _workbook_packet_descriptors,
        workbook_packet_commitments,
    ) = _validate_reviewer_workbooks(
        packet_paths=reviewer_workbook_packet_paths,
        reviewer_workbook_paths=reviewer_workbook_paths,
        target_packet=packet,
        target_packet_sha256=_sha256(packet_payload),
        protocol_path=protocol_path,
        codebook_path=codebook_path,
        secret=secret,
    )
    if len(reviewer_ledger_paths) != 2:
        _fail("resolver workload requires exactly two reviewer ledgers")
    reviewers: list[str] = []
    decisions_by_slot: dict[str, dict[str, dict[str, Any]]] = {}
    ledger_descriptors: dict[str, dict[str, Any]] = {}
    for index, path in enumerate(reviewer_ledger_paths):
        path = Path(path).resolve()
        payload = _read(path)
        ledger = _json_payload(payload, context=f"reviewer ledger {index}")
        slot = ledger.get("reviewer_slot")
        if slot not in assignments or slot in decisions_by_slot:
            _fail("reviewer ledgers must cover assignments A and B exactly once")
        pseudonym, parsed = _validate_reviewer_ledger(
            ledger,
            packet=packet,
            packet_sha256=_sha256(packet_payload),
            protocol_sha256=protocol_descriptor["sha256"],
            codebook_sha256=codebook_descriptor["sha256"],
            dimensions=dimensions,
            expected_assignment=assignments[slot],
        )
        reviewers.append(pseudonym)
        decisions_by_slot[slot] = parsed
        ledger_descriptors[slot] = _descriptor(path, payload)
    if set(decisions_by_slot) != {"A", "B"}:
        _fail("reviewer ledgers must cover assignments A and B exactly once")
    if reviewers[0].casefold() == reviewers[1].casefold():
        _fail("the two reviewer pseudonyms must differ")
    decisions = [decisions_by_slot["A"], decisions_by_slot["B"]]
    disagreement_ids = {
        case["case_id"]
        for case in cases
        if _decision_signature(decisions[0][case["case_id"]])
        != _decision_signature(decisions[1][case["case_id"]])
    }
    if not disagreement_ids:
        return {"status": "not_required", "disagreement_count": 0}
    reviewer_descriptors = [ledger_descriptors["A"], ledger_descriptors["B"]]
    workload_cases = [
        {
            "case_id": case["case_id"],
            "material": case["material"],
            "reviewer_decisions": [
                decisions[0][case["case_id"]],
                decisions[1][case["case_id"]],
            ],
        }
        for case in cases
        if case["case_id"] in disagreement_ids
    ]
    unsigned: dict[str, Any] = {
        "artifact_kind": RESOLVER_WORKLOAD_KIND,
        "review_phase": "full",
        "packet_id": packet["packet_id"],
        "protocol_sha256": protocol_descriptor["sha256"],
        "packet_sha256": _sha256(packet_payload),
        "codebook_sha256": codebook_descriptor["sha256"],
        "reviewer_ledger_sha256s": [
            descriptor["sha256"] for descriptor in reviewer_descriptors
        ],
        "reviewer_assignments": [assignments["A"], assignments["B"]],
        "reviewer_workbook_sha256s": [
            descriptor["sha256"] for descriptor in workbook_descriptors
        ],
        "reviewer_workbook_packets": workbook_packet_commitments,
        "cases": workload_cases,
    }
    workload_id = "workload_" + _sha256(_canonical_bytes(unsigned))
    workload = {**unsigned, "workload_id": workload_id}
    workload_payload = _canonical_bytes(workload, pretty=True)
    output_dir = Path(output_dir).resolve()
    workload_path = output_dir / f"{workload_id}.json"
    workbook_payload = _resolver_html(
        workload,
        workload_sha256=_sha256(workload_payload),
        codebook=_public_codebook(codebook),
    )
    workbook_path = output_dir / f"resolver_{workload_id}.html"
    _atomic_write(workload_path, workload_payload, private=False)
    _atomic_write(workbook_path, workbook_payload, private=False)
    return {
        "status": "ready",
        "resolver_workload": workload_path,
        "resolver_workload_sha256": _sha256(workload_payload),
        "resolver_workbook": workbook_path,
        "resolver_workbook_sha256": _sha256(workbook_payload),
        "disagreement_count": len(workload_cases),
        "agreement_count_excluded": len(cases) - len(workload_cases),
    }

def _validate_resolver(
    *,
    workload_path: Path,
    resolver_workbook_path: Path,
    resolver_ledger_path: Path,
    packet: Mapping[str, Any],
    packet_sha256: str,
    protocol_sha256: str,
    codebook_sha256: str,
    codebook: Mapping[str, Any],
    dimensions: frozenset[str],
    reviewer_ledger_sha256s: list[str],
    reviewer_decisions: Sequence[Mapping[str, Mapping[str, Any]]],
    reviewer_assignments: list[dict[str, Any]],
    reviewer_workbook_sha256s: list[str],
    reviewer_workbook_packet_commitments: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    workload_path = Path(workload_path).resolve()
    workload_payload = _read(workload_path)
    workload = _json_payload(workload_payload, context="resolver workload")
    required_workload = {
        "artifact_kind",
        "review_phase",
        "packet_id",
        "protocol_sha256",
        "packet_sha256",
        "codebook_sha256",
        "reviewer_ledger_sha256s",
        "reviewer_assignments",
        "reviewer_workbook_sha256s",
        "reviewer_workbook_packets",
        "cases",
        "workload_id",
    }
    if (
        not isinstance(workload, Mapping)
        or set(workload) != required_workload
        or workload.get("artifact_kind") != RESOLVER_WORKLOAD_KIND
    ):
        _fail("resolver workload schema or artifact kind differs")
    unsigned = dict(workload)
    workload_id = unsigned.pop("workload_id")
    if workload_id != "workload_" + _sha256(_canonical_bytes(unsigned)):
        _fail("resolver workload content binding differs")
    expected_bindings = {
        "review_phase": "full",
        "packet_id": packet["packet_id"],
        "protocol_sha256": protocol_sha256,
        "packet_sha256": packet_sha256,
        "codebook_sha256": codebook_sha256,
        "reviewer_ledger_sha256s": reviewer_ledger_sha256s,
        "reviewer_assignments": reviewer_assignments,
        "reviewer_workbook_sha256s": reviewer_workbook_sha256s,
        "reviewer_workbook_packets": reviewer_workbook_packet_commitments,
    }
    for field, expected in expected_bindings.items():
        if workload.get(field) != expected:
            _fail(f"resolver workload {field} binding differs")
    all_cases = {case["case_id"]: case for case in packet["cases"]}
    disagreements = {
        case_id
        for case_id in all_cases
        if _decision_signature(reviewer_decisions[0][case_id])
        != _decision_signature(reviewer_decisions[1][case_id])
    }
    expected_cases = [
        {
            "case_id": case["case_id"],
            "material": case["material"],
            "reviewer_decisions": [
                reviewer_decisions[0][case["case_id"]],
                reviewer_decisions[1][case["case_id"]],
            ],
        }
        for case in packet["cases"]
        if case["case_id"] in disagreements
    ]
    if workload.get("cases") != expected_cases or not disagreements:
        _fail("resolver workload must contain every disagreement and no agreement")

    resolver_workbook_path = Path(resolver_workbook_path).resolve()
    resolver_workbook_payload = _read(resolver_workbook_path)
    expected_resolver_workbook = _resolver_html(
        workload,
        workload_sha256=_sha256(workload_payload),
        codebook=_public_codebook(codebook),
    )
    if resolver_workbook_payload != expected_resolver_workbook:
        _fail("resolver workbook is not the exact canonical HTML")

    ledger_path = Path(resolver_ledger_path).resolve()
    ledger_payload = _read(ledger_path)
    ledger = _json_payload(ledger_payload, context="resolver ledger")
    required_ledger = {
        "artifact_kind",
        "role",
        "review_phase",
        "packet_id",
        "protocol_sha256",
        "packet_sha256",
        "codebook_sha256",
        "resolver_workload_sha256",
        "reviewer_ledger_sha256s",
        "resolver_pseudonym",
        "human_attestation",
        "started_at",
        "completed_at",
        "decisions",
    }
    if (
        not isinstance(ledger, Mapping)
        or set(ledger) != required_ledger
        or ledger.get("artifact_kind") != LEDGER_KIND
        or ledger.get("role") != "resolver"
    ):
        _fail("resolver ledger schema, artifact kind, or role differs")
    for field in (
        "review_phase",
        "packet_id",
        "protocol_sha256",
        "packet_sha256",
        "codebook_sha256",
        "reviewer_ledger_sha256s",
    ):
        if ledger.get(field) != expected_bindings[field]:
            _fail(f"resolver ledger {field} binding differs")
    if ledger.get("resolver_workload_sha256") != _sha256(workload_payload):
        _fail("resolver ledger workload binding differs")
    pseudonym = ledger.get("resolver_pseudonym")
    if not isinstance(pseudonym, str) or PSEUDONYM.fullmatch(pseudonym) is None:
        _fail("resolver pseudonym is invalid")
    if ledger.get("human_attestation") != HUMAN_ATTESTATION:
        _fail("resolver ledger lacks the required human-only attestation")
    started = _timestamp(ledger.get("started_at"), "resolver ledger started_at")
    completed = _timestamp(ledger.get("completed_at"), "resolver ledger completed_at")
    if datetime.fromisoformat(completed.replace("Z", "+00:00")) < datetime.fromisoformat(
        started.replace("Z", "+00:00")
    ):
        _fail("resolver ledger completion precedes its start")
    raw_decisions = ledger.get("decisions")
    if not isinstance(raw_decisions, list):
        _fail("resolver decisions must be an array")
    expected_resolution_order = [
        case["case_id"]
        for case in packet["cases"]
        if case["case_id"] in disagreements
    ]
    if [row.get("case_id") if isinstance(row, Mapping) else None for row in raw_decisions] != expected_resolution_order:
        _fail("resolver decisions are not in exact disagreement order")
    resolutions: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_decisions):
        parsed = _decision(
            raw,
            context=f"resolver decisions[{index}]",
            case_ids=disagreements,
            dimensions=dimensions,
        )
        if parsed["case_id"] in resolutions:
            _fail("resolver ledger repeats a disagreement")
        resolutions[parsed["case_id"]] = parsed
    if set(resolutions) != disagreements:
        _fail("resolver ledger must cover every disagreement and no agreement")
    return (
        pseudonym,
        resolutions,
        dict(workload),
        dict(ledger),
        _descriptor(resolver_workbook_path, resolver_workbook_payload),
    )

def _summary(count: int, total: int) -> dict[str, Any]:
    if (
        isinstance(count, bool)
        or isinstance(total, bool)
        or not isinstance(count, int)
        or not isinstance(total, int)
        or not 0 <= count <= total
    ):
        _fail("summary count and denominator are inconsistent")
    return {
        "count": count,
        "denominator": total,
        "proportion": None if total == 0 else count / total,
    }


def _cohen_kappa(
    first: Mapping[str, Mapping[str, Any]],
    second: Mapping[str, Mapping[str, Any]],
    case_order: Sequence[str],
) -> dict[str, Any]:
    matrix = {
        left: {right: 0 for right in CLASSIFICATIONS}
        for left in CLASSIFICATIONS
    }
    for case_id in case_order:
        matrix[first[case_id]["classification"]][second[case_id]["classification"]] += 1
    total = len(case_order)
    agreements = sum(matrix[name][name] for name in CLASSIFICATIONS)
    observed = None if total == 0 else agreements / total
    expected: float | None
    kappa: float | None
    if total == 0:
        expected = None
        kappa = None
    else:
        first_marginals = {
            name: sum(matrix[name].values()) / total for name in CLASSIFICATIONS
        }
        second_marginals = {
            name: sum(matrix[left][name] for left in CLASSIFICATIONS) / total
            for name in CLASSIFICATIONS
        }
        expected = sum(
            first_marginals[name] * second_marginals[name]
            for name in CLASSIFICATIONS
        )
        kappa = None if expected == 1.0 else (observed - expected) / (1.0 - expected)
    return {
        "classes": list(CLASSIFICATIONS),
        "confusion_matrix_a_by_b": matrix,
        "observed_agreement": observed,
        "expected_chance_agreement": expected,
        "cohen_kappa": kappa,
    }


def _derived_judgment(error_type: str, classification: str) -> str:
    if error_type == "false_positive":
        return "defensible" if classification in {"supports_claim", "indeterminate"} else "non_defensible"
    if error_type == "false_negative":
        return "defensible" if classification in {"rejects_claim", "indeterminate"} else "non_defensible"
    _fail("administrator error type differs")


def adjudicate_review(
    *,
    packet_path: Path,
    admin_manifest_path: Path,
    protocol_path: Path,
    codebook_path: Path,
    reviewer_ledger_paths: Sequence[Path],
    reviewer_workbook_packet_paths: Sequence[Path],
    reviewer_workbook_paths: Sequence[Path],
    blinding_key: bytes,
    resolver_workload_path: Path | None = None,
    resolver_workbook_path: Path | None = None,
    resolver_ledger_path: Path | None = None,
) -> dict[str, Any]:
    """Validate a complete human chain and summarize the full threshold-error census."""

    secret = _key(blinding_key)
    _protocol, protocol_descriptor = _validate_protocol(protocol_path)
    codebook, codebook_descriptor, dimensions = _validate_codebook(
        codebook_path, protocol_sha256=protocol_descriptor["sha256"]
    )
    if codebook["status"] != "frozen":
        _fail("scientific adjudication requires a frozen codebook")
    _validate_frozen_codebook_authenticity(codebook, secret=secret)

    packet_path = Path(packet_path).resolve()
    packet_payload = _read(packet_path)
    packet = _json_payload(packet_payload, context="full review packet")
    _packet_id, cases = _validate_packet(
        packet,
        protocol_sha256=protocol_descriptor["sha256"],
        codebook_sha256=codebook_descriptor["sha256"],
        secret=secret,
    )
    if packet["review_phase"] != "full":
        _fail("pilot decisions cannot be published as a scientific error review")

    (
        assignments,
        workbook_descriptors,
        workbook_packet_descriptors,
        workbook_packet_commitments,
    ) = _validate_reviewer_workbooks(
        packet_paths=reviewer_workbook_packet_paths,
        reviewer_workbook_paths=reviewer_workbook_paths,
        target_packet=packet,
        target_packet_sha256=_sha256(packet_payload),
        protocol_path=protocol_path,
        codebook_path=codebook_path,
        secret=secret,
    )

    admin_path = Path(admin_manifest_path).resolve()
    admin_payload = _read(admin_path)
    admin = _json_payload(admin_payload, context="error-review admin manifest")
    _validate_admin(
        admin,
        packet_path=packet_path,
        packet=packet,
        packet_payload=packet_payload,
        protocol_descriptor=protocol_descriptor,
        codebook_descriptor=codebook_descriptor,
        secret=secret,
    )

    if len(reviewer_ledger_paths) != 2:
        _fail("exactly two independent reviewer ledgers are required")
    reviewer_names_by_slot: dict[str, str] = {}
    reviewer_decisions_by_slot: dict[str, dict[str, dict[str, Any]]] = {}
    reviewer_descriptors_by_slot: dict[str, dict[str, Any]] = {}
    reviewer_comments: list[dict[str, Any]] = []
    for index, path in enumerate(reviewer_ledger_paths):
        path = Path(path).resolve()
        payload = _read(path)
        ledger = _json_payload(payload, context=f"reviewer ledger {index}")
        slot = ledger.get("reviewer_slot")
        if slot not in assignments or slot in reviewer_decisions_by_slot:
            _fail("reviewer ledgers must cover assignments A and B exactly once")
        pseudonym, decisions = _validate_reviewer_ledger(
            ledger,
            packet=packet,
            packet_sha256=_sha256(packet_payload),
            protocol_sha256=protocol_descriptor["sha256"],
            codebook_sha256=codebook_descriptor["sha256"],
            dimensions=dimensions,
            expected_assignment=assignments[slot],
        )
        reviewer_names_by_slot[slot] = pseudonym
        reviewer_decisions_by_slot[slot] = decisions
        reviewer_descriptors_by_slot[slot] = _descriptor(path, payload)
        for case_id, decision in decisions.items():
            if decision["comment"]:
                reviewer_comments.append(
                    {
                        "case_id": case_id,
                        "source": f"reviewer_{slot}",
                        "classification": decision["classification"],
                        "dimensions": decision["dimensions"],
                        "comment": decision["comment"],
                    }
                )
    if set(reviewer_decisions_by_slot) != {"A", "B"}:
        _fail("reviewer ledgers must cover assignments A and B exactly once")
    if (
        reviewer_names_by_slot["A"].casefold()
        == reviewer_names_by_slot["B"].casefold()
    ):
        _fail("the two reviewer pseudonyms must differ")

    reviewer_names = [reviewer_names_by_slot["A"], reviewer_names_by_slot["B"]]
    reviewer_decisions = [
        reviewer_decisions_by_slot["A"],
        reviewer_decisions_by_slot["B"],
    ]
    reviewer_descriptors = [
        reviewer_descriptors_by_slot["A"],
        reviewer_descriptors_by_slot["B"],
    ]
    disagreements = {
        case["case_id"]
        for case in cases
        if _decision_signature(reviewer_decisions[0][case["case_id"]])
        != _decision_signature(reviewer_decisions[1][case["case_id"]])
    }

    resolver_values = (
        resolver_workload_path,
        resolver_workbook_path,
        resolver_ledger_path,
    )
    if any(value is not None for value in resolver_values) and not all(
        value is not None for value in resolver_values
    ):
        _fail("resolver workload, workbook, and ledger must be supplied together")
    if disagreements and not all(value is not None for value in resolver_values):
        _fail(
            "human resolver required: reviewer disagreements cannot be synthesized or left unresolved"
        )
    if not disagreements and any(value is not None for value in resolver_values):
        _fail("resolver artifacts are forbidden when reviewers fully agree")

    resolver_name: str | None = None
    resolutions: dict[str, dict[str, Any]] = {}
    resolver_workbook_descriptor: dict[str, Any] | None = None
    if disagreements:
        assert resolver_workload_path is not None
        assert resolver_workbook_path is not None
        assert resolver_ledger_path is not None
        (
            resolver_name,
            resolutions,
            _resolver_workload,
            _resolver_ledger,
            resolver_workbook_descriptor,
        ) = _validate_resolver(
            workload_path=resolver_workload_path,
            resolver_workbook_path=resolver_workbook_path,
            resolver_ledger_path=resolver_ledger_path,
            packet=packet,
            packet_sha256=_sha256(packet_payload),
            protocol_sha256=protocol_descriptor["sha256"],
            codebook_sha256=codebook_descriptor["sha256"],
            codebook=codebook,
            dimensions=dimensions,
            reviewer_ledger_sha256s=[
                row["sha256"] for row in reviewer_descriptors
            ],
            reviewer_decisions=reviewer_decisions,
            reviewer_assignments=[assignments["A"], assignments["B"]],
            reviewer_workbook_sha256s=[
                row["sha256"] for row in workbook_descriptors
            ],
            reviewer_workbook_packet_commitments=workbook_packet_commitments,
        )
        if resolver_name.casefold() in {name.casefold() for name in reviewer_names}:
            _fail("resolver pseudonym must differ from both reviewer pseudonyms")
        for case_id, decision in resolutions.items():
            if decision["comment"]:
                reviewer_comments.append(
                    {
                        "case_id": case_id,
                        "source": "resolver",
                        "classification": decision["classification"],
                        "dimensions": decision["dimensions"],
                        "comment": decision["comment"],
                    }
                )

    admin_by_case = {
        row["case_id"]: row for row in admin["case_mapping"]
    }
    final: list[dict[str, Any]] = []
    judgment_counts = {"defensible": 0, "non_defensible": 0}
    classification_counts = {name: 0 for name in CLASSIFICATIONS}
    error_type_counts = {
        "false_positive": {"defensible": 0, "non_defensible": 0},
        "false_negative": {"defensible": 0, "non_defensible": 0},
    }
    dimension_counts: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = case["case_id"]
        if case_id in disagreements:
            chosen = resolutions[case_id]
            source = "resolver"
        else:
            chosen = reviewer_decisions[0][case_id]
            source = "reviewer_agreement"
        error_type = admin_by_case[case_id]["error_type"]
        classification = chosen["classification"]
        judgment = _derived_judgment(error_type, classification)
        classification_counts[classification] += 1
        judgment_counts[judgment] += 1
        error_type_counts[error_type][judgment] += 1
        for dimension in chosen["dimensions"]:
            bucket = dimension_counts.setdefault(
                dimension,
                {
                    "by_judgment": {"defensible": 0, "non_defensible": 0},
                    "by_error_type": {"false_positive": 0, "false_negative": 0},
                },
            )
            bucket["by_judgment"][judgment] += 1
            bucket["by_error_type"][error_type] += 1
        final.append(
            {
                "case_id": case_id,
                "error_type": error_type,
                "human_classification": classification,
                "judgment": judgment,
                "defensibility_basis": (
                    "indeterminate_ambiguity"
                    if classification == "indeterminate"
                    else (
                        "human_matches_system"
                        if judgment == "defensible"
                        else "human_matches_reference"
                    )
                ),
                "dimensions": chosen["dimensions"],
                "comment": chosen["comment"],
                "decision_source": source,
            }
        )

    total = len(cases)
    dimension_rows = []
    for dimension in sorted(dimension_counts):
        bucket = dimension_counts[dimension]
        count = sum(bucket["by_judgment"].values())
        dimension_rows.append(
            {
                "dimension": dimension,
                **_summary(count, total),
                "by_judgment": bucket["by_judgment"],
                "by_error_type": bucket["by_error_type"],
            }
        )
    error_type_summaries: dict[str, Any] = {}
    for error_type in ("false_positive", "false_negative"):
        stratum_total = sum(error_type_counts[error_type].values())
        error_type_summaries[error_type] = {
            **_summary(stratum_total, total),
            "defensible": _summary(
                error_type_counts[error_type]["defensible"], stratum_total
            ),
            "non_defensible": _summary(
                error_type_counts[error_type]["non_defensible"], stratum_total
            ),
        }

    provenance: dict[str, Any] = {
        "protocol": _public_commitment(protocol_descriptor),
        "codebook": _public_commitment(codebook_descriptor),
        "packet": _public_commitment(_descriptor(packet_path, packet_payload)),
        "admin_manifest": _public_commitment(_descriptor(admin_path, admin_payload)),
        "reviewer_ledgers": _public_commitment(reviewer_descriptors),
        "reviewer_workbooks": _public_commitment(workbook_descriptors),
        "reviewer_assignments": [assignments["A"], assignments["B"]],
        "reviewer_workbook_packets": _public_commitment(
            workbook_packet_descriptors
        ),
        "comparison_inputs": _public_commitment(admin["provenance"]),
        "resolver_workload": None,
        "resolver_workbook": None,
        "resolver_ledger": None,
    }
    if disagreements:
        assert resolver_workload_path is not None
        assert resolver_ledger_path is not None
        provenance["resolver_workload"] = _public_commitment(
            _descriptor(resolver_workload_path)
        )
        provenance["resolver_workbook"] = _public_commitment(
            resolver_workbook_descriptor
        )
        provenance["resolver_ledger"] = _public_commitment(
            _descriptor(resolver_ledger_path)
        )

    case_order = [case["case_id"] for case in cases]
    return {
        "artifact_kind": REPORT_KIND,
        "status": "complete",
        "panel_id": admin["provenance"]["panel_id"],
        "arm_id": admin["provenance"]["arm_id"],
        "model_id": admin["provenance"]["model_id"],
        "packet_id": packet["packet_id"],
        "evaluated_statements": packet["evaluated_statement_count"],
        "threshold_errors": _summary(total, packet["evaluated_statement_count"]),
        "error_types": error_type_summaries,
        "human_classifications": {
            name: _summary(classification_counts[name], total)
            for name in CLASSIFICATIONS
        },
        "review": {
            "reviewer_pseudonyms": reviewer_names,
            "resolver_pseudonym": resolver_name,
            "exact_agreement": _summary(total - len(disagreements), total),
            "disagreement_count": len(disagreements),
            "resolved_by_resolver_count": len(resolutions),
            "classification_reliability": _cohen_kappa(
                reviewer_decisions[0], reviewer_decisions[1], case_order
            ),
            "human_attestation": HUMAN_ATTESTATION,
        },
        "defensibility": {
            "denominator": "all_threshold_errors",
            "defensible": _summary(judgment_counts["defensible"], total),
            "non_defensible": _summary(
                judgment_counts["non_defensible"], total
            ),
            "system_supported_defensible": _summary(
                judgment_counts["defensible"]
                - classification_counts["indeterminate"],
                total,
            ),
            "indeterminate_ambiguity_defensible": _summary(
                classification_counts["indeterminate"], total
            ),
            "unresolved": _summary(0, total),
        },
        "dimensions": {
            "multiple_dimensions_per_case": True,
            "denominator": "all_threshold_errors",
            "rows": dimension_rows,
        },
        "taxonomy_refinements": reviewer_comments,
        "adjudications": final,
        "provenance": provenance,
    }
