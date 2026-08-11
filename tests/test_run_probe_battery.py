"""Pure contract tests for the A2 MLX probe-battery runner."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_probe_battery.py"

_spec = importlib.util.spec_from_file_location("run_probe_battery", SCRIPT)
runner = importlib.util.module_from_spec(_spec)
sys.modules["run_probe_battery"] = runner
assert _spec.loader is not None
_spec.loader.exec_module(runner)


def _fit_row(*, row_index: int = 7) -> dict[str, object]:
    return {
        "subject": "MAPK1",
        "object": "JUN",
        "stmt_type": "Activation",
        "evidence_text": "MAPK1 activates JUN in stimulated cells.",
        "row_index": row_index,
        "source_hash": "123",
        "pa_hash": "456",
        "tag": "correct",
        "gold_correct": True,
        "matches_hash": "789",
        "source_api": "reach",
    }


def _probe_value(
    *,
    probe_id: str | None = None,
    argmax_token_id: int | None = None,
    secs: float = 0.25,
) -> dict[str, object]:
    probe = runner.probe_by_id(probe_id or runner.BASE_PROBE_ID)
    return runner.probe_record(
        probe=probe,
        log_p_correct=-0.4,
        log_p_incorrect=-1.4,
        argmax_token_id=(
            runner.LABEL_TOKEN_IDS[0]
            if argmax_token_id is None
            else argmax_token_id
        ),
        secs=secs,
    )


def _artifact_parts(*, n_rows: int = 1):
    probe = runner.probe_by_id(runner.BASE_PROBE_ID)
    rows = [_fit_row(row_index=index) for index in range(n_rows)]
    manifest = runner.build_manifest(
        gold_path=ROOT / "data" / "benchmark" / "eval_curation_v1.jsonl",
        split="fit",
        model=runner.DEFAULT_MODEL,
        rows=rows,
        evidence_source="evidence_text",
        selection={"mode": "all", "n": n_rows, "seed": None},
        probes=(probe,),
        started_at="2026-08-09T00:00:00+00:00",
    )
    records = [
        runner.wide_record(
            row=row,
            probe_values={probe.id: _probe_value()},
            elapsed_s=4.0,
        )
        for row in rows
    ]
    return manifest, records


def _write_jsonl(path: Path, values) -> None:
    path.write_text(
        "".join(json.dumps(value, allow_nan=False) + "\n" for value in values),
        encoding="utf-8",
    )


def test_top_level_ast_has_no_optional_model_runtime_imports():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    imported_roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots.isdisjoint({"mlx", "mlx_lm", "transformers"})


def test_template_geometry_and_score_one_contract_without_mlx(monkeypatch):
    class FakeScalar:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    class FakeLogprobs:
        def __getitem__(self, token_id):
            return FakeScalar(
                {
                    runner.LABEL_TOKEN_IDS[0]: -0.25,
                    runner.LABEL_TOKEN_IDS[1]: -1.75,
                }[token_id]
            )

    class FakeTokenizer:
        def __init__(self):
            self.encode_calls = []
            self.template_calls = []

        def encode(self, text, *, add_special_tokens):
            self.encode_calls.append((text, add_special_tokens))
            if text == "correct":
                return [runner.LABEL_TOKEN_IDS[0]]
            if text == "incorrect":
                return [runner.LABEL_TOKEN_IDS[1]]
            return [10, 20, 30]

        def apply_chat_template(self, messages, **kwargs):
            self.template_calls.append((messages, kwargs))
            return "<bos>chat<|channel>thought\n<channel|>"

    tokenizer = FakeTokenizer()
    runner._assert_template_geometry(tokenizer)
    assert tokenizer.encode_calls[:2] == [
        ("correct", False),
        ("incorrect", False),
    ]
    assert tokenizer.template_calls[0][1] == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }

    calls = {}
    mlx_package = types.ModuleType("mlx")
    mlx_package.__path__ = []
    mlx_core = types.ModuleType("mlx.core")
    mlx_core.array = lambda values: ("array", values)
    mlx_core.argmax = lambda values: FakeScalar(7)
    mlx_package.core = mlx_core
    mlx_lm_package = types.ModuleType("mlx_lm")
    mlx_lm_package.__path__ = []
    mlx_generate = types.ModuleType("mlx_lm.generate")

    def fake_generate_step(ids, model, *, max_tokens):
        calls.update(ids=ids, model=model, max_tokens=max_tokens)
        yield FakeScalar(999999), FakeLogprobs()

    mlx_generate.generate_step = fake_generate_step
    monkeypatch.setitem(sys.modules, "mlx", mlx_package)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_package)
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", mlx_generate)

    model = object()
    result = runner._score_one(
        model,
        tokenizer,
        "rendered prompt",
        runner.LABEL_TOKEN_IDS,
    )

    assert result == (-0.25, -1.75, 7)
    assert calls == {
        "ids": ("array", [10, 20, 30]),
        "model": model,
        "max_tokens": 1,
    }
    assert tokenizer.encode_calls[-1] == ("rendered prompt", False)


def test_runner_never_imports_or_calls_oriented_p():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "oriented_p" not in imported_names
    assert "oriented_p" not in called_names


def test_real_brace_bearing_evidence_renders_verbatim():
    raw_rows = runner._read_jsonl(
        ROOT / "data" / "benchmark" / "eval_curation_v1.jsonl"
    )
    brace_indices = [
        index
        for index, row in enumerate(raw_rows)
        if "{" in row["evidence_text"] or "}" in row["evidence_text"]
    ]
    assert brace_indices == [521, 644, 682, 1234, 1321]

    probe = runner.probe_by_id(runner.BASE_PROBE_ID)
    for index in brace_indices:
        evidence = raw_rows[index]["evidence_text"]
        _system, user, _prefill = runner.render(probe, raw_rows[index])
        assert evidence in user
        assert "{" in user or "}" in user


def test_holdout_join_copies_only_evidence_text(monkeypatch, tmp_path):
    benchmark_path = tmp_path / "belief_benchmark.jsonl"
    evidence = "Joined evidence {with braces}."
    _write_jsonl(
        benchmark_path,
        [
            {
                "source_hash": 42,
                "evidence_text": evidence,
                "subject": "WRONG SUBJECT",
                "object": "WRONG OBJECT",
                "stmt_type": "WrongType",
                "tag": "incorrect",
                "pa_hash": "unsafe-pa",
                "matches_hash": "unsafe-matches",
                "source_api": "unsafe-api",
            },
            {
                "source_hash": "42",
                "evidence_text": evidence,
                "subject": "ANOTHER WRONG SUBJECT",
                "object": "ANOTHER WRONG OBJECT",
                "stmt_type": "AnotherWrongType",
                "tag": "grounding",
                "pa_hash": "another-unsafe-pa",
            },
        ],
    )
    monkeypatch.setattr(runner, "BELIEF_BENCHMARK_PATH", benchmark_path)
    holdout = {
        "source_hash": 42,
        "subject": "HOLDOUT SUBJECT",
        "object": "HOLDOUT OBJECT",
        "stmt_type": "Activation",
        "tag": "correct",
    }

    joined = runner.join_holdout_evidence([holdout])

    assert len(joined) == 1
    row = joined[0]
    assert tuple(row) == runner.NORMALIZED_ROW_FIELDS
    assert row["evidence_text"] == evidence
    assert row["subject"] == holdout["subject"]
    assert row["object"] == holdout["object"]
    assert row["stmt_type"] == holdout["stmt_type"]
    assert row["tag"] == holdout["tag"]
    assert row["source_hash"] == "42"
    assert row["pa_hash"] is None
    assert row["matches_hash"] is None
    assert row["source_api"] is None
    assert row["gold_correct"] is True


def test_probe_record_preserves_exact_delta_and_probability_logit():
    probe = runner.probe_by_id(runner.BASE_PROBE_ID)
    log_p_correct = -2.25
    log_p_incorrect = -3.5

    record = runner.probe_record(
        probe=probe,
        log_p_correct=log_p_correct,
        log_p_incorrect=log_p_incorrect,
        argmax_token_id=runner.LABEL_TOKEN_IDS[0],
        secs=0.125,
    )

    assert record["delta_logit"] == log_p_correct - log_p_incorrect
    p_raw = record["p_raw"]
    assert p_raw is not None
    assert abs(math.log(p_raw / (1.0 - p_raw)) - record["delta_logit"]) < 1e-9
    assert record["both_observed"] is True
    assert record["precision_limited"] is False


def test_probe_record_survives_label_mass_underflow():
    probe = runner.probe_by_id(runner.BASE_PROBE_ID)
    record = runner.probe_record(
        probe=probe,
        log_p_correct=-800.0,
        log_p_incorrect=-801.0,
        argmax_token_id=runner.LABEL_TOKEN_IDS[0],
        secs=None,
    )

    assert record["p_raw"] is None
    assert record["status"] == "no_label_mass"
    assert record["delta_logit"] == pytest.approx(1.0)
    assert record["log_label_mass"] == pytest.approx(
        -800.0 + math.log1p(math.exp(-1.0))
    )
    assert record["secs"] is None


def test_probe_record_flags_a_non_label_argmax():
    assert _probe_value(argmax_token_id=0)["argmax_is_label"] is False


def test_parse_args_defaults_and_repeatable_probe_id(tmp_path):
    required = [
        "--gold",
        str(tmp_path / "gold.jsonl"),
        "--split",
        "fit",
        "--out",
        str(tmp_path / "out.jsonl"),
    ]
    defaults = runner.parse_args(required)
    assert defaults.model == runner.DEFAULT_MODEL
    assert defaults.probe_id == list(runner.PROBE_IDS)
    assert defaults.seed == 0
    assert not hasattr(defaults, "limit")
    assert defaults.sample is None

    with pytest.raises(SystemExit):
        runner.parse_args(required + ["--limit", "1"])

    repeated = runner.parse_args(
        required
        + [
            "--probe-id",
            runner.BASE_PROBE_ID,
            "--probe-id",
            "tax.relation_present",
        ]
    )
    assert repeated.probe_id == [runner.BASE_PROBE_ID, "tax.relation_present"]


def test_single_class_cli_refuses_without_flag_and_passes_with_it(
    tmp_path, capsys
):
    gold_path = tmp_path / "one_class.jsonl"
    _write_jsonl(
        gold_path,
        [
            {
                "pa_hash": 1,
                "source_hash": 2,
                "matches_hash": 3,
                "stmt_type": "Activation",
                "subject": "A",
                "object": "B",
                "evidence_text": "A activates B.",
                "source_api": "reach",
                "tag": "correct",
            }
        ],
    )
    out_path = tmp_path / "dry"
    argv = [
        "--gold",
        str(gold_path),
        "--split",
        "fit",
        "--out",
        str(out_path),
        "--probe-id",
        runner.BASE_PROBE_ID,
        "--dry-run",
    ]

    assert runner.main(argv) == 1
    assert "--sample N --seed S" in capsys.readouterr().err
    assert runner.main(argv + ["--allow-single-class"]) == 0
    assert Path(f"{out_path}.prompts.jsonl").is_file()


def test_verify_artifact_passes_clean_and_rejects_duplicate_row_index(
    tmp_path, capsys
):
    clean_path = tmp_path / "clean.jsonl"
    manifest, records = _artifact_parts(n_rows=1)
    _write_jsonl(clean_path, [manifest, *records])

    assert runner.main(["--verify-artifact", str(clean_path)]) == 0
    output = capsys.readouterr().out
    assert "manifest_lines=1" in output
    assert "n_records=1" in output
    assert "n_distinct_row_index=1" in output
    assert "argmax_is_label_rate=1.000000" in output
    assert 'status_histogram={"ok":1}' in output
    assert "n_probe_records=1" in output
    assert "median_s_per_record=4.000000" in output
    assert "median_s_per_probe_read=0.250000" in output
    assert "missing_secs=0" in output
    assert "missing_elapsed_s=0" in output

    duplicate_path = tmp_path / "duplicate.jsonl"
    duplicate_manifest, duplicate_records = _artifact_parts(n_rows=2)
    duplicate_records[1]["row_index"] = duplicate_records[0]["row_index"]
    _write_jsonl(duplicate_path, [duplicate_manifest, *duplicate_records])

    assert runner.main(["--verify-artifact", str(duplicate_path)]) == 1
    assert "duplicated row_index" in capsys.readouterr().err


def test_verify_reports_record_and_probe_read_timings_separately(
    tmp_path, capsys
):
    probe_ids = (runner.BASE_PROBE_ID, "tax.relation_present")
    probes = tuple(runner.probe_by_id(probe_id) for probe_id in probe_ids)
    row = _fit_row(row_index=0)
    manifest = runner.build_manifest(
        gold_path=ROOT / "data" / "benchmark" / "eval_curation_v1.jsonl",
        split="fit",
        model=runner.DEFAULT_MODEL,
        rows=[row],
        evidence_source="evidence_text",
        selection={"mode": "all", "n": 1, "seed": None},
        probes=probes,
        started_at="2026-08-09T00:00:00+00:00",
    )
    probe_values = {
        probe_ids[0]: _probe_value(probe_id=probe_ids[0], secs=0.1),
        probe_ids[1]: _probe_value(probe_id=probe_ids[1], secs=0.5),
    }
    record = runner.wide_record(
        row=row,
        probe_values=probe_values,
        elapsed_s=3.0,
    )
    artifact = tmp_path / "separate_timings.jsonl"
    _write_jsonl(artifact, [manifest, record])

    assert runner.main(["--verify-artifact", str(artifact)]) == 0
    output = capsys.readouterr().out
    assert "n_records=1" in output
    assert "n_probe_records=2" in output
    assert "median_s_per_record=3.000000" in output
    assert "median_s_per_probe_read=0.300000" in output
    assert "median_s_per_record=0.300000" not in output


def test_verifier_rejects_nonfinite_and_inconsistent_probe_values():
    manifest, records = _artifact_parts(n_rows=1)
    probe_id = runner.BASE_PROBE_ID

    nonfinite = copy.deepcopy(records)
    nonfinite[0]["probes"][probe_id]["log_p_correct"] = math.inf
    with pytest.raises(runner.ArtifactError, match="log_p_correct is not finite"):
        runner._validate_artifact_parts(manifest, nonfinite, require_complete=True)

    inconsistent_delta = copy.deepcopy(records)
    inconsistent_delta[0]["probes"][probe_id]["delta_logit"] = 99.0
    with pytest.raises(runner.ArtifactError, match="delta_logit does not match"):
        runner._validate_artifact_parts(
            manifest, inconsistent_delta, require_complete=True
        )

    inconsistent_argmax = copy.deepcopy(records)
    inconsistent_argmax[0]["probes"][probe_id]["argmax_token_id"] = 0
    with pytest.raises(runner.ArtifactError, match="argmax_is_label does not match"):
        runner._validate_artifact_parts(
            manifest, inconsistent_argmax, require_complete=True
        )

    inconsistent_probability = copy.deepcopy(records)
    inconsistent_probability[0]["probes"][probe_id]["status"] = "no_label_mass"
    with pytest.raises(runner.ArtifactError, match="status does not match"):
        runner._validate_artifact_parts(
            manifest, inconsistent_probability, require_complete=True
        )


def test_verifier_rejects_empty_or_incoherent_selection_manifest():
    empty_manifest, empty_records = _artifact_parts(n_rows=0)
    with pytest.raises(runner.ArtifactError, match="n_rows must be a positive"):
        runner._validate_artifact_parts(
            empty_manifest, empty_records, require_complete=True
        )

    manifest, records = _artifact_parts(n_rows=1)
    bad_selection = copy.deepcopy(manifest)
    bad_selection["selection"]["n"] = 2
    with pytest.raises(runner.ArtifactError, match="selection n must equal"):
        runner._validate_artifact_parts(
            bad_selection, records, require_complete=True
        )

    legacy_limit = copy.deepcopy(manifest)
    legacy_limit["selection"]["mode"] = "limit"
    with pytest.raises(runner.ArtifactError, match="mode must be all or sample"):
        runner._validate_artifact_parts(
            legacy_limit, records, require_complete=True
        )


def test_resume_rejects_same_ordinal_with_changed_provenance(tmp_path):
    manifest, records = _artifact_parts(n_rows=1)
    records[0]["source_hash"] = "stale-source"
    artifact = tmp_path / "stale.jsonl"
    _write_jsonl(artifact, [manifest, *records])

    with pytest.raises(runner.ArtifactError, match="provenance mismatch for source_hash"):
        runner._resume_rows(
            artifact,
            expected_manifest=manifest,
            expected_rows=[_fit_row(row_index=0)],
        )


def test_resume_appends_after_unterminated_complete_json_line(
    monkeypatch, tmp_path
):
    manifest, records = _artifact_parts(n_rows=2)
    artifact = tmp_path / "partial.jsonl"
    artifact.write_text(
        runner._json_line(manifest) + "\n" + runner._json_line(records[0]),
        encoding="utf-8",
    )

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return "chat"

    monkeypatch.setattr(
        runner, "_load_model", lambda model_id: (object(), FakeTokenizer())
    )
    monkeypatch.setattr(runner, "_assert_template_geometry", lambda tok: None)
    monkeypatch.setattr(
        runner,
        "_score_one",
        lambda model, tok, prompt, label_ids: (-0.4, -1.4, label_ids[0]),
    )
    rows = [_fit_row(row_index=index) for index in range(2)]
    probe = runner.probe_by_id(runner.BASE_PROBE_ID)

    assert runner.run_scoring(
        out_path=artifact,
        manifest=manifest,
        rows=rows,
        probes=(probe,),
        model_id=runner.DEFAULT_MODEL,
        resume=True,
    ) == 0

    lines = artifact.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert all(isinstance(json.loads(line), dict) for line in lines)
    summary = runner._validate_artifact_parts(
        *runner._parse_artifact(artifact), require_complete=True
    )
    assert summary["n_records"] == 2


def test_scoring_rejects_substrate_hint_even_without_dry_run(
    monkeypatch, tmp_path
):
    row = _fit_row(row_index=0)
    row["evidence_text"] = "SUBSTRATE HINT: forbidden phase-specific text"
    probe = runner.probe_by_id(runner.BASE_PROBE_ID)
    manifest = runner.build_manifest(
        gold_path=ROOT / "data" / "benchmark" / "eval_curation_v1.jsonl",
        split="fit",
        model=runner.DEFAULT_MODEL,
        rows=[row],
        evidence_source="evidence_text",
        selection={"mode": "all", "n": 1, "seed": None},
        probes=(probe,),
        started_at="2026-08-09T00:00:00+00:00",
    )
    monkeypatch.setattr(runner, "_load_model", lambda model_id: (object(), object()))
    monkeypatch.setattr(runner, "_assert_template_geometry", lambda tok: None)
    monkeypatch.setattr(
        runner,
        "_score_one",
        lambda *args: pytest.fail("scoring must not run for a substrate-hint prompt"),
    )

    with pytest.raises(ValueError, match="contains SUBSTRATE HINT"):
        runner.run_scoring(
            out_path=tmp_path / "forbidden.jsonl",
            manifest=manifest,
            rows=[row],
            probes=(probe,),
            model_id=runner.DEFAULT_MODEL,
            resume=False,
        )


def test_exactly_one_probe_is_declared_base():
    meta = runner._probe_meta(runner.PROBES)
    assert set(meta) == set(runner.PROBE_IDS)
    assert sum(value["is_base"] for value in meta.values()) == 1
    assert meta[runner.BASE_PROBE_ID]["is_base"] is True


def test_emitted_wide_row_satisfies_every_c1_input_field():
    manifest, records = _artifact_parts(n_rows=1)
    row = records[0]

    assert manifest["_manifest"] is True
    assert isinstance(manifest["probe_meta"], dict)
    assert sum(
        value["is_base"] for value in manifest["probe_meta"].values()
    ) == 1
    for value in manifest["probe_meta"].values():
        assert isinstance(value["family"], str)
        assert isinstance(value["is_base"], bool)

    assert isinstance(row["row_index"], int)
    assert isinstance(row["source_hash"], str)
    assert isinstance(row["pa_hash"], str)
    assert isinstance(row["probes"], dict)
    assert isinstance(row["elapsed_s"], float)
    for probe_value in row["probes"].values():
        assert {"p_raw", "status", "both_observed", "precision_limited"} <= set(
            probe_value
        )
    assert "row_i" not in row
    assert all("lp_status" not in value for value in row["probes"].values())


def test_sample_selection_is_seeded_and_keeps_gold_ordinals():
    rows = [{**_fit_row(row_index=index), "gold_correct": bool(index % 2)} for index in range(8)]
    selected_a, declaration_a = runner.select_rows(rows, sample=4, seed=11)
    selected_b, declaration_b = runner.select_rows(rows, sample=4, seed=11)
    assert [row["row_index"] for row in selected_a] == [
        row["row_index"] for row in selected_b
    ]
    assert declaration_a == declaration_b == {"mode": "sample", "n": 4, "seed": 11}
    assert [row["row_index"] for row in selected_a] != list(range(4))
    assert [row["row_index"] for row in rows] == list(range(8))
