"""Probe pipeline.

Replaces the monolithic parse_evidence with four narrow probes
(subject_role, object_role, relation_axis, scope) routed through a
substrate-first orchestrator. Each probe commits a single decision from
a closed answer set; the adjudicator combines them via a flat decision
table.

Why four narrow probes rather than one call: a single parse produced answers
that drifted between runs because nothing pinned WHICH question was being
answered. A probe that commits one decision from a closed answer set is
checkable; a free-form parse is not. The adjudicator's decision table is flat
and deterministic so a disagreement is attributable to a named probe.
(The S-phase doctrine record that argued this in full was removed from the
tree; it is in git history.)
"""
from indra_belief.scorers.probes import (
    object_role,
    relation_axis,
    scope,
    subject_role,
)
from indra_belief.scorers.probes.router import substrate_route
from indra_belief.scorers.probes.types import (
    ObjectRoleAnswer,
    PerturbationMarker,
    ProbeBundle,
    ProbeConfidence,
    ProbeKind,
    ProbeRequest,
    ProbeResponse,
    ProbeSource,
    RelationAxisAnswer,
    ScopeAnswer,
    SubjectRoleAnswer,
)

__all__ = [
    "ObjectRoleAnswer",
    "PerturbationMarker",
    "ProbeBundle",
    "ProbeConfidence",
    "ProbeKind",
    "ProbeRequest",
    "ProbeResponse",
    "ProbeSource",
    "RelationAxisAnswer",
    "ScopeAnswer",
    "SubjectRoleAnswer",
    "object_role",
    "relation_axis",
    "scope",
    "subject_role",
    "substrate_route",
]
