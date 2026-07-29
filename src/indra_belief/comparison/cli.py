"""Single command-line entry point for the INDRA belief comparison."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
from typing import Any, Mapping, Sequence

from indra_belief.model_client import ModelClient

from . import assemble, error_review, llm, metrics, report
from .contracts import canonical_json_line, load_run_plan, strict_json_loads
from .inputs import load_inputs
from .runner import (
    RunnerError,
    inspect_plan,
    prepare_run,
    read_token_fd,
    run_prepared,
    write_ready_fd,
)


DEFAULT_PLAN = Path("data/comparison/run_plan.json")
DEFAULT_INPUTS = Path("data/comparison/inputs.json")
DEFAULT_SPEC = Path("data/results/indra_belief_comparison_spec.json")
DEFAULT_METRICS = Path("data/results/indra_belief_comparison_metrics.json")
DEFAULT_LITERATURE = Path("data/benchmark/indra_paper_2023_published_method_metrics.json")
DEFAULT_MARKDOWN = Path("reports/indra_belief_comparison.md")
DEFAULT_HTML = Path("reports/indra_belief_comparison.html")
DEFAULT_REPORT_MANIFEST = Path("reports/indra_belief_comparison_manifest.json")
TOKEN_ENV = "AWS_BEARER_TOKEN_BEDROCK"


def _model_bundle_arguments(
    *, inputs_path: Path, plan_path: Path, action_id: str
) -> dict[str, Any]:
    """Resolve one bundle entirely from the frozen plan and model declaration."""

    inputs_path = inputs_path.resolve()
    inputs = load_inputs(inputs_path)
    plan = load_run_plan(plan_path)
    action = plan.action_by_id.get(action_id)
    if action is None:
        raise ValueError(f"unknown model-bundle action {action_id!r}")
    declarations = [model for model in inputs.llm_models if model.action_id == action_id]
    if len(declarations) != 1:
        raise ValueError(
            f"model-bundle action {action_id!r} must have exactly one LLM declaration"
        )
    declaration = declarations[0]
    stage = plan.stage_by_id[action.stage_id]
    if (
        declaration.run_id != action.run_id
        or declaration.served_model != stage.model
        or declaration.provider_model_id != stage.provider_model_id
    ):
        raise ValueError(
            f"model-bundle action {action_id!r} disagrees with its frozen LLM declaration"
        )

    replay_capture = plan.replay_manifest.capture(context="replay manifest")
    replay = strict_json_loads(replay_capture.payload, context="replay manifest")
    if not isinstance(replay, Mapping):
        raise ValueError("replay manifest must be an object")
    workloads = replay.get("workloads")
    if not isinstance(workloads, list):
        raise ValueError("replay manifest workloads must be an array")
    matches = [
        item
        for item in workloads
        if isinstance(item, Mapping) and item.get("name") == action.workload
    ]
    if len(matches) != 1:
        raise ValueError(
            f"replay manifest must declare workload {action.workload!r} exactly once"
        )
    workload = matches[0]

    def workload_path(field: str) -> Path:
        descriptor = workload.get(field)
        if not isinstance(descriptor, Mapping) or not isinstance(
            descriptor.get("path"), str
        ):
            raise ValueError(f"replay workload {action.workload!r} lacks {field}")
        declared = Path(descriptor["path"])
        path = (declared if declared.is_absolute() else plan.root / declared).resolve()
        try:
            path.relative_to(plan.root.resolve())
        except ValueError as exc:
            raise ValueError(f"replay workload {field} escapes the repository") from exc
        return path

    comparison_dir = inputs_path.parent
    return {
        "run_plan": plan.capture,
        "raw_attempts": action.output,
        "execution_map": workload_path("execution_map"),
        "statements": workload_path("corpus"),
        "spend_ledger": action.ledger,
        "aggregation": comparison_dir / "aggregation.json",
        "pricing": comparison_dir / "pricing.json",
        "output_dir": declaration.bundle_manifest.parent,
        "run_id": action.run_id,
        "served_model": stage.model,
        "model_id": declaration.model_id,
        "provider_model_id": stage.provider_model_id,
        "workload": action.workload,
    }


def _print_json(value: Mapping[str, Any]) -> None:
    sys.stdout.buffer.write(canonical_json_line(value))
    sys.stdout.buffer.flush()


def _print_complete(**outputs: Path) -> None:
    _print_json(
        {
            "status": "complete",
            "outputs": {name: str(path) for name, path in outputs.items()},
        }
    )


def _print_result(value: Mapping[str, Any]) -> None:
    def convert(item: Any) -> Any:
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, Mapping):
            return {key: convert(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(child) for child in item]
        return item

    _print_json(convert(value))


def _dotenv_token(path: Path) -> str | None:
    if not path.is_file():
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if separator and key.strip() == TOKEN_ENV:
            token = value.strip()
            if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
                token = token[1:-1]
            return token or None
    return None


def _bearer_token(root: Path) -> str:
    token = os.environ.get(TOKEN_ENV) or _dotenv_token(root / ".env")
    if not token or "\n" in token or "\r" in token:
        raise RunnerError(f"{TOKEN_ENV} is absent or malformed")
    if len(token.encode("utf-8")) > 16_384:
        raise RunnerError(f"{TOKEN_ENV} exceeds the control-channel bound")
    return token


def _read_readiness(descriptor: int, process: subprocess.Popen[Any], timeout: float = 1800.0) -> dict[str, Any]:
    # Preflight revalidates every arm's append-only ledger byte-for-byte
    # before the token is read; at hundreds of thousands of raw rows that
    # takes minutes, so this pre-credential liveness bound must scale with
    # ledger growth. No provider call can happen before readiness.
    selector = selectors.DefaultSelector()
    selector.register(descriptor, selectors.EVENT_READ)
    chunks: list[bytes] = []
    total = 0
    try:
        while True:
            events = selector.select(timeout)
            if not events:
                raise RunnerError("credential-free child did not become ready in time")
            chunk = os.read(descriptor, 8192)
            if not chunk:
                raise RunnerError(
                    f"credential-free child exited before readiness (status {process.poll()})"
                )
            chunks.append(chunk)
            total += len(chunk)
            if total > 65_536:
                raise RunnerError("readiness message exceeds 64 KiB")
            if b"\n" in chunk:
                break
    finally:
        selector.close()
        os.close(descriptor)
    raw = b"".join(chunks)
    line, separator, remainder = raw.partition(b"\n")
    if not separator or remainder:
        raise RunnerError("readiness channel contains more than one JSON line")
    value = strict_json_loads(line, context="readiness")
    if not isinstance(value, dict) or value.get("status") != "ready_for_bearer_token":
        raise RunnerError("child readiness message is invalid")
    if value.get("token_read") is not False or value.get("provider_calls_started_during_preflight") != 0:
        raise RunnerError("child did not preserve the pre-credential boundary")
    return value


def _terminate(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _bearer_after_readiness(plan: Any, readiness: Mapping[str, Any]) -> str:
    if readiness.get("plan_sha256") != plan.sha256:
        raise RunnerError("child readiness belongs to a different run plan")
    return _bearer_token(plan.root)


def _run_parent(plan_path: Path, action_id: str | None) -> int:
    """Run one action with the bearer withheld from the validating child."""

    plan = load_run_plan(plan_path)
    ready_read, ready_write = os.pipe()
    token_read, token_write = os.pipe()
    command = [
        sys.executable,
        "-m",
        "indra_belief.comparison",
        "_run-child",
        "--plan",
        str(plan.path),
        "--ready-fd",
        str(ready_write),
        "--token-fd",
        str(token_read),
    ]
    if action_id:
        command.extend(["--action", action_id])
    environment = os.environ.copy()
    for key in (
        TOKEN_ENV,
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
    ):
        environment.pop(key, None)
    process = subprocess.Popen(
        command,
        cwd=plan.root,
        env=environment,
        pass_fds=(ready_write, token_read),
        start_new_session=True,
    )
    os.close(ready_write)
    os.close(token_read)
    try:
        readiness = _read_readiness(ready_read, process)
        _print_json(readiness)
        token = _bearer_after_readiness(plan, readiness)
        payload = token.encode("utf-8") + b"\n"
        token = ""
        written = os.write(token_write, payload)
        expected = len(payload)
        payload = b""
        if written != expected:
            raise RunnerError("bearer-token control write failed")
        os.close(token_write)
        token_write = -1
        return process.wait()
    except BaseException:
        _terminate(process)
        raise
    finally:
        if token_write >= 0:
            os.close(token_write)


def _run_child(plan_path: Path, action_id: str | None, ready_fd: int, token_fd: int) -> int:
    prepared = prepare_run(plan_path, action_id=action_id)

    def client_factory(token: str, _action: Any) -> ModelClient:
        return ModelClient(prepared.stage.model, bedrock_bearer_token=token)

    summary = run_prepared(
        prepared,
        ready_writer=lambda value: write_ready_fd(ready_fd, value),
        token_reader=lambda: read_token_fd(token_fd),
        client_factory=client_factory,
    )
    _print_json(summary.as_dict())
    return 0 if summary.status == "complete" else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)

    status = commands.add_parser("status", help="validate inputs and show ordered action state")
    status.add_argument("--plan", type=Path, default=DEFAULT_PLAN)

    run = commands.add_parser("run", help="run one dependency-ready paid action")
    run.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    run.add_argument("--action", help="optional dependency-ready action to run")

    child = commands.add_parser("_run-child", help=argparse.SUPPRESS)
    child.add_argument("--plan", type=Path, required=True)
    child.add_argument("--action")
    child.add_argument("--ready-fd", type=int, required=True)
    child.add_argument("--token-fd", type=int, required=True)

    bundle = commands.add_parser("model-bundle", help="materialize one completed LLM run")
    bundle.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    bundle.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    bundle.add_argument("--action", required=True)

    materialize = commands.add_parser("materialize", help="assemble the shared-gold metric specification")
    materialize.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    materialize.add_argument("--output", type=Path, default=DEFAULT_SPEC)
    materialize.add_argument("--force", action="store_true")

    metric = commands.add_parser("metrics", help="compute paired metrics, calibration, cost, and Pareto")
    metric.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    metric.add_argument("--output", type=Path, default=DEFAULT_METRICS)
    metric.add_argument("--force", action="store_true")

    rendered = commands.add_parser("report", help="render Markdown and HTML from the metrics artifact")
    rendered.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    rendered.add_argument("--literature", type=Path, default=DEFAULT_LITERATURE)
    rendered.add_argument(
        "--error-review",
        type=Path,
        action="append",
        default=[],
        help="completed panel review; provide once for each canonical paper panel",
    )
    rendered.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    rendered.add_argument("--html", type=Path, default=DEFAULT_HTML)
    rendered.add_argument("--manifest", type=Path, default=DEFAULT_REPORT_MANIFEST)

    review_key = commands.add_parser(
        "error-review-key", help="create the administrator-held review blinding key"
    )
    review_key.add_argument(
        "--output", type=Path, default=Path("data/comparison/error_review.key")
    )

    review_codebook = commands.add_parser(
        "error-review-codebook-create",
        help="create the human-pilot dimension codebook (does not claim a pilot occurred)",
    )
    review_codebook.add_argument(
        "--protocol", type=Path, default=Path("data/comparison/error_review.json")
    )
    review_codebook.add_argument("--output", type=Path, required=True)

    review_freeze = commands.add_parser(
        "error-review-codebook-freeze",
        help="freeze a human-refined codebook after two complete pilot reviews",
    )
    review_freeze.add_argument(
        "--protocol", type=Path, default=Path("data/comparison/error_review.json")
    )
    review_freeze.add_argument("--pilot-codebook", type=Path, required=True)
    review_freeze.add_argument("--candidate-codebook", type=Path, required=True)
    review_freeze.add_argument("--pilot-packet", type=Path, required=True)
    review_freeze.add_argument("--pilot-admin-manifest", type=Path, required=True)
    review_freeze.add_argument(
        "--pilot-workbooks", type=Path, nargs=2, required=True,
        metavar=("WORKBOOK_A", "WORKBOOK_B"),
    )
    review_freeze.add_argument("--blinding-key-file", type=Path, required=True)
    review_freeze.add_argument(
        "--reviews", type=Path, nargs=2, required=True, metavar=("REVIEW_A", "REVIEW_B")
    )
    review_freeze.add_argument("--frozen-at", required=True)
    review_freeze.add_argument("--attest-human-freeze", action="store_true")
    review_freeze.add_argument("--output", type=Path, required=True)

    review_prepare = commands.add_parser(
        "error-review-prepare",
        help="derive an opaque packet and private mapping from one exact LLM bundle panel",
    )
    review_prepare.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    review_prepare.add_argument("--bundle", type=Path, required=True)
    review_prepare.add_argument("--panel", required=True)
    review_prepare.add_argument("--arm", required=True)
    review_prepare.add_argument(
        "--protocol", type=Path, default=Path("data/comparison/error_review.json")
    )
    review_prepare.add_argument("--codebook", type=Path, required=True)
    review_prepare.add_argument("--blinding-key-file", type=Path, required=True)
    review_prepare.add_argument(
        "--reviewer-output-dir", type=Path,
        default=Path("data/comparison/error_reviews"),
    )
    review_prepare.add_argument("--admin-output-dir", type=Path, required=True)
    review_prepare.add_argument("--pilot-case-count", type=int)
    review_prepare.add_argument("--created-at")

    review_workbook = commands.add_parser(
        "error-review-workbook",
        help="build one self-contained deduplicating offline workbook from opaque packets",
    )
    review_workbook.add_argument("--packets", type=Path, nargs="+", required=True)
    review_workbook.add_argument(
        "--protocol", type=Path, default=Path("data/comparison/error_review.json")
    )
    review_workbook.add_argument("--codebook", type=Path, required=True)
    review_workbook.add_argument("--blinding-key-file", type=Path, required=True)
    review_workbook.add_argument(
        "--output-dir", type=Path, default=Path("data/comparison/error_reviews")
    )

    review_resolver = commands.add_parser(
        "error-review-resolver",
        help="build the complete disagreement-only resolver workload and offline workbook",
    )
    review_resolver.add_argument("--packet", type=Path, required=True)
    review_resolver.add_argument(
        "--protocol", type=Path, default=Path("data/comparison/error_review.json")
    )
    review_resolver.add_argument("--codebook", type=Path, required=True)
    review_resolver.add_argument("--blinding-key-file", type=Path, required=True)
    review_resolver.add_argument(
        "--reviews", type=Path, nargs=2, required=True, metavar=("REVIEW_A", "REVIEW_B")
    )
    review_resolver.add_argument(
        "--workbook-packets", type=Path, nargs="+", required=True,
        help="the exact ordered packet list used to generate the reviewer workbook",
    )
    review_resolver.add_argument(
        "--reviewer-workbooks", type=Path, nargs=2, required=True,
        metavar=("WORKBOOK_A", "WORKBOOK_B"),
    )
    review_resolver.add_argument(
        "--output-dir", type=Path, default=Path("data/comparison/error_reviews")
    )

    review_adjudicate = commands.add_parser(
        "error-review-adjudicate",
        help="validate and summarize one full two-reviewer human error review",
    )
    review_adjudicate.add_argument("--packet", type=Path, required=True)
    review_adjudicate.add_argument("--admin-manifest", type=Path, required=True)
    review_adjudicate.add_argument(
        "--protocol", type=Path, default=Path("data/comparison/error_review.json")
    )
    review_adjudicate.add_argument("--codebook", type=Path, required=True)
    review_adjudicate.add_argument("--blinding-key-file", type=Path, required=True)
    review_adjudicate.add_argument(
        "--reviews", type=Path, nargs=2, required=True, metavar=("REVIEW_A", "REVIEW_B")
    )
    review_adjudicate.add_argument(
        "--workbook-packets", type=Path, nargs="+", required=True,
        help="the exact ordered packet list used to generate the reviewer workbook",
    )
    review_adjudicate.add_argument(
        "--reviewer-workbooks", type=Path, nargs=2, required=True,
        metavar=("WORKBOOK_A", "WORKBOOK_B"),
    )
    review_adjudicate.add_argument("--resolver-workload", type=Path)
    review_adjudicate.add_argument("--resolver-workbook", type=Path)
    review_adjudicate.add_argument("--resolver-ledger", type=Path)
    review_adjudicate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "status":
            _print_json(inspect_plan(args.plan).as_dict())
            return 0
        if args.command == "run":
            return _run_parent(args.plan, args.action)
        if args.command == "_run-child":
            return _run_child(args.plan, args.action, args.ready_fd, args.token_fd)
        if args.command == "model-bundle":
            arguments = _model_bundle_arguments(
                inputs_path=args.inputs,
                plan_path=args.plan,
                action_id=args.action,
            )
            llm.materialize_model_bundle(**arguments)
            _print_complete(manifest=arguments["output_dir"] / "manifest.json")
            return 0
        if args.command == "materialize":
            inputs = load_inputs(args.inputs)
            spec = assemble.assemble_spec(inputs, args.output)
            assemble.write_spec(spec, args.output, force=args.force)
            _print_complete(spec=args.output)
            return 0
        if args.command == "metrics":
            artifact = metrics.build_artifact(args.spec)
            metrics.write_artifact(artifact, args.output, force=args.force)
            _print_complete(metrics=args.output)
            return 0
        if args.command == "report":
            report.render_reports(
                args.metrics,
                markdown_path=args.markdown,
                html_path=args.html,
                manifest_path=args.manifest,
                literature_path=args.literature,
                error_review_paths=args.error_review,
            )
            _print_complete(
                markdown=args.markdown,
                html=args.html,
                manifest=args.manifest,
            )
            return 0
        if args.command == "error-review-key":
            error_review.generate_blinding_key(args.output)
            _print_complete(blinding_key=args.output)
            return 0
        if args.command == "error-review-codebook-create":
            value = error_review.make_pilot_codebook(args.protocol)
            error_review.write_json(value, args.output)
            _print_complete(codebook=args.output)
            return 0
        if args.command == "error-review-codebook-freeze":
            value = error_review.freeze_codebook(
                protocol_path=args.protocol,
                pilot_codebook_path=args.pilot_codebook,
                candidate_codebook_path=args.candidate_codebook,
                pilot_packet_path=args.pilot_packet,
                pilot_admin_manifest_path=args.pilot_admin_manifest,
                pilot_workbook_paths=args.pilot_workbooks,
                reviewer_ledger_paths=args.reviews,
                blinding_key=error_review.load_blinding_key(args.blinding_key_file),
                human_freeze_attested=args.attest_human_freeze,
                frozen_at=args.frozen_at,
                output_path=args.output,
            )
            _print_result(value)
            return 0
        if args.command == "error-review-prepare":
            value = error_review.prepare_review_artifacts(
                spec_path=args.spec,
                bundle_manifest_path=args.bundle,
                panel_id=args.panel,
                arm_id=args.arm,
                protocol_path=args.protocol,
                codebook_path=args.codebook,
                blinding_key=error_review.load_blinding_key(args.blinding_key_file),
                reviewer_output_dir=args.reviewer_output_dir,
                admin_output_dir=args.admin_output_dir,
                pilot_case_count=args.pilot_case_count,
                created_at=args.created_at,
            )
            _print_result(value)
            return 0
        if args.command == "error-review-workbook":
            value = error_review.generate_reviewer_workbook(
                packet_paths=args.packets,
                protocol_path=args.protocol,
                codebook_path=args.codebook,
                blinding_key=error_review.load_blinding_key(args.blinding_key_file),
                output_dir=args.output_dir,
            )
            _print_result(value)
            return 0
        if args.command == "error-review-resolver":
            value = error_review.generate_resolver_workload(
                packet_path=args.packet,
                protocol_path=args.protocol,
                codebook_path=args.codebook,
                reviewer_ledger_paths=args.reviews,
                reviewer_workbook_packet_paths=args.workbook_packets,
                reviewer_workbook_paths=args.reviewer_workbooks,
                blinding_key=error_review.load_blinding_key(args.blinding_key_file),
                output_dir=args.output_dir,
            )
            _print_result(value)
            return 0
        if args.command == "error-review-adjudicate":
            value = error_review.adjudicate_review(
                packet_path=args.packet,
                admin_manifest_path=args.admin_manifest,
                protocol_path=args.protocol,
                codebook_path=args.codebook,
                blinding_key=error_review.load_blinding_key(args.blinding_key_file),
                reviewer_ledger_paths=args.reviews,
                reviewer_workbook_packet_paths=args.workbook_packets,
                reviewer_workbook_paths=args.reviewer_workbooks,
                resolver_workload_path=args.resolver_workload,
                resolver_workbook_path=args.resolver_workbook,
                resolver_ledger_path=args.resolver_ledger,
            )
            error_review.write_json(value, args.output)
            _print_complete(error_review=args.output)
            return 0
    except (ValueError, OSError, llm.LlmMaterializationError) as exc:
        parser = _parser()
        parser.error(str(exc))
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
