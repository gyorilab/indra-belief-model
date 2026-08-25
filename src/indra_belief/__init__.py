"""Evidence-quality scoring for INDRA biomedical text-mining extractions.

Public API (the monolithic scorer — the only one, and the one that was
empirically dominant on holdout_cc, F1 0.751 vs the retired decomposed 0.657):
    score_statement(statement, client) -> list[dict]
        Score every evidence in an INDRA Statement. Mirrors INDRA's
        abstraction (a Statement owns a list of Evidence objects);
        returns one per-sentence verdict per evidence.
    score_evidence(statement, evidence, client) -> dict
        Score a single (Statement, Evidence) pair — the atomic unit.
        Use when you want to judge one evidence sentence without
        iterating the rest of the Statement's evidence list.
    ModelClient(model_name)
        Backend-agnostic transport (OpenAI-compatible, Anthropic).

The decomposed four-probe path that used to sit beside this one has been
removed; see README for usage examples.
"""
__all__ = [
    "score_statement",
    "score_evidence",
    "ModelClient",
    "ModelResponse",
]


def __getattr__(name: str):
    """Load public conveniences lazily.

    Importing a transport or spend-accounting submodule must not implicitly
    import INDRA, Gilda, the ontology, or the monolithic scorer.  The previous
    eager package imports made that impossible even for otherwise stdlib-only
    paid transports.  Public ``from indra_belief import ...`` behavior is
    preserved while submodule imports now pay only for their own closure.
    """
    if name in {"ModelClient", "ModelResponse"}:
        from indra_belief.model_client import ModelClient, ModelResponse

        return {"ModelClient": ModelClient, "ModelResponse": ModelResponse}[name]
    if name in {"score_evidence", "score_statement"}:
        from indra_belief.scorers.monolithic import score_evidence, score_statement

        return {
            "score_evidence": score_evidence,
            "score_statement": score_statement,
        }[name]
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
