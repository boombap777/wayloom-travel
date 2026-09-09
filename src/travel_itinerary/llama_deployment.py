"""Reproducible local llama.cpp CLI deployment and smoke benchmark.

This module intentionally benchmarks one-shot ``llama-cli`` invocations.  Its wall-clock
latency therefore includes model loading and is not comparable to a persistent HTTP server's
hot-request latency.  The report also distinguishes the raw quantized model contract from a
mechanical tool-argument repair: the repair only copies an already-complete ``request`` into
``tool_call.arguments`` and never fills a missing user constraint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import load_jsonl
from .model_evaluation import extract_first_json_object, validate_decision_payload
from .model_output import repair_missing_tool_arguments

_PERF_PATTERN = re.compile(
    r"\[\s*Prompt:\s*(?P<prompt>[0-9.]+)\s*t/s\s*\|\s*"
    r"Generation:\s*(?P<generation>[0-9.]+)\s*t/s\s*\]"
)


@dataclass(frozen=True, slots=True)
class LlamaCppDeploymentConfig:
    config_path: Path
    project_root: Path
    version: str
    deployment_id: str
    base_gguf: Path
    lora_gguf: Path
    system_prompt: Path
    frozen_evaluation: Path
    case_ids: tuple[str, ...]
    context_size: int
    n_predict: int
    gpu_layers: int
    temperature: float
    seed: int
    timeout_seconds: int
    report_path: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_deployment_config(config_path: Path) -> LlamaCppDeploymentConfig:
    config_path = config_path.resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = config_path.parents[2]
    runtime = raw["runtime"]
    config = LlamaCppDeploymentConfig(
        config_path=config_path,
        project_root=project_root,
        version=str(raw["version"]),
        deployment_id=str(raw["deployment_id"]),
        base_gguf=_resolve_path(project_root, raw["base_gguf"]),
        lora_gguf=_resolve_path(project_root, raw["lora_gguf"]),
        system_prompt=_resolve_path(project_root, raw["system_prompt"]),
        frozen_evaluation=_resolve_path(project_root, raw["frozen_evaluation"]),
        case_ids=tuple(raw["case_ids"]),
        context_size=int(runtime["context_size"]),
        n_predict=int(runtime["n_predict"]),
        gpu_layers=int(runtime["gpu_layers"]),
        temperature=float(runtime["temperature"]),
        seed=int(runtime["seed"]),
        timeout_seconds=int(runtime["timeout_seconds"]),
        report_path=_resolve_path(project_root, raw["report_path"]),
    )
    if not config.case_ids:
        raise ValueError("case_ids must not be empty")
    if config.context_size <= 0 or config.n_predict <= 0 or config.timeout_seconds <= 0:
        raise ValueError("context_size, n_predict, and timeout_seconds must be positive")
    for artifact in (
        config.base_gguf,
        config.lora_gguf,
        config.system_prompt,
        config.frozen_evaluation,
    ):
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
    return config


def resolve_llama_cli(explicit_path: Path | None = None) -> Path:
    candidate = explicit_path or (
        Path(os.environ["TRAVEL_LLAMA_CLI_PATH"])
        if os.environ.get("TRAVEL_LLAMA_CLI_PATH")
        else None
    )
    if candidate is None:
        raise RuntimeError("set TRAVEL_LLAMA_CLI_PATH or pass --llama-cli")
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def build_cli_command(
    config: LlamaCppDeploymentConfig,
    llama_cli: Path,
    case: dict[str, Any],
) -> list[str]:
    prompt = f"当前日期：{case['today']}\n用户请求：{case['message']}"
    return [
        str(llama_cli),
        "-m",
        str(config.base_gguf),
        "--lora",
        str(config.lora_gguf),
        "-ngl",
        str(config.gpu_layers),
        "-c",
        str(config.context_size),
        "-n",
        str(config.n_predict),
        "--temp",
        str(config.temperature),
        "--seed",
        str(config.seed),
        "--no-warmup",
        "--no-display-prompt",
        "--single-turn",
        "--reasoning",
        "off",
        "--simple-io",
        "--log-disable",
        "-sysf",
        str(config.system_prompt),
        "-p",
        prompt,
    ]


def parse_cli_perf(stdout: str) -> dict[str, float | None]:
    match = _PERF_PATTERN.search(stdout)
    if match is None:
        return {"prompt_tokens_per_second": None, "generation_tokens_per_second": None}
    return {
        "prompt_tokens_per_second": float(match.group("prompt")),
        "generation_tokens_per_second": float(match.group("generation")),
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 3)


def _selected_cases(config: LlamaCppDeploymentConfig) -> list[dict[str, Any]]:
    by_id = {case["id"]: case for case in load_jsonl(config.frozen_evaluation)}
    missing = [case_id for case_id in config.case_ids if case_id not in by_id]
    if missing:
        raise ValueError(f"configured case ids are absent from frozen evaluation: {missing}")
    return [by_id[case_id] for case_id in config.case_ids]


def run_cli_case(
    config: LlamaCppDeploymentConfig,
    llama_cli: Path,
    case: dict[str, Any],
) -> dict[str, Any]:
    command = build_cli_command(config, llama_cli, case)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=config.project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=config.timeout_seconds,
        check=False,
    )
    wall_latency_ms = round((time.perf_counter() - started) * 1000, 3)
    parsed_raw = extract_first_json_object(completed.stdout)
    raw_decision, raw_contract_error = validate_decision_payload(parsed_raw)
    repaired_payload, repair_notes = repair_missing_tool_arguments(parsed_raw)
    repaired_decision, repaired_contract_error = validate_decision_payload(repaired_payload)
    perf = parse_cli_perf(completed.stdout)
    expected_arguments = case.get("expected_tool_arguments")
    actual_arguments = (
        repaired_decision.get("tool_call", {}).get("arguments") if repaired_decision else None
    )
    return {
        "id": case["id"],
        "expected_action": case["expected_action"],
        "exit_code": completed.returncode,
        "wall_latency_ms": wall_latency_ms,
        **perf,
        "raw_json_valid": parsed_raw is not None,
        "raw_contract_valid": raw_decision is not None,
        "raw_contract_error": raw_contract_error,
        "repaired_contract_valid": repaired_decision is not None,
        "repaired_contract_error": repaired_contract_error,
        "repair_notes": repair_notes,
        "action_match_after_repair": (
            repaired_decision is not None
            and repaired_decision["action"] == case["expected_action"]
        ),
        "tool_arguments_match_after_repair": (
            actual_arguments == expected_arguments if expected_arguments is not None else None
        ),
        "raw_model_json": parsed_raw,
        "repaired_decision": repaired_decision,
        "raw_cli_stdout": completed.stdout,
        "raw_cli_stderr": completed.stderr,
    }


def _build_metrics(results: list[dict[str, Any]]) -> dict[str, float | int | None]:
    count = len(results)
    wall_latencies = [float(result["wall_latency_ms"]) for result in results]
    generation_speeds = [
        float(result["generation_tokens_per_second"])
        for result in results
        if result["generation_tokens_per_second"] is not None
    ]
    tool_results = [
        result for result in results if result["tool_arguments_match_after_repair"] is not None
    ]
    return {
        "case_count": count,
        "raw_json_valid_rate": sum(result["raw_json_valid"] for result in results) / count,
        "raw_contract_valid_rate": sum(result["raw_contract_valid"] for result in results) / count,
        "contract_valid_rate_after_repair": sum(
            result["repaired_contract_valid"] for result in results
        )
        / count,
        "action_exact_match_after_repair": sum(
            result["action_match_after_repair"] for result in results
        )
        / count,
        "tool_argument_exact_match_after_repair": (
            sum(result["tool_arguments_match_after_repair"] for result in tool_results)
            / len(tool_results)
            if tool_results
            else None
        ),
        "mechanical_tool_argument_repair_count": sum(bool(result["repair_notes"]) for result in results),
        "one_shot_wall_latency_ms_p50": _percentile(wall_latencies, 0.5),
        "one_shot_wall_latency_ms_p95": _percentile(wall_latencies, 0.95),
        "one_shot_wall_latency_ms_mean": round(statistics.fmean(wall_latencies), 3),
        "generation_tokens_per_second_p50": _percentile(generation_speeds, 0.5),
        "generation_tokens_per_second_p95": _percentile(generation_speeds, 0.95),
    }


def _llama_version(llama_cli: Path) -> str:
    completed = subprocess.run(
        [str(llama_cli), "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return (completed.stdout or completed.stderr).strip()


def run_deployment_benchmark(
    config: LlamaCppDeploymentConfig,
    *,
    llama_cli: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    cases = _selected_cases(config)
    results = [run_cli_case(config, llama_cli, case) for case in cases]
    output_path = output_path or config.report_path
    report = {
        "report_version": "travel-llama-cpp-deploy-v1",
        "deployment_id": config.deployment_id,
        "config_version": config.version,
        "measurement_scope": (
            "Three selected cases from the versioned frozen split. Each result is a one-shot "
            "llama-cli process invocation, so wall latency includes process startup and model load."
        ),
        "repair_policy": (
            "Only a missing tool_call.arguments object may be rebuilt from an already-complete "
            "model request. No missing slot is inferred or filled."
        ),
        "artifacts": {
            "base_gguf": str(config.base_gguf),
            "base_gguf_sha256": _sha256_file(config.base_gguf),
            "lora_gguf": str(config.lora_gguf),
            "lora_gguf_sha256": _sha256_file(config.lora_gguf),
            "system_prompt": str(config.system_prompt),
            "system_prompt_sha256": _sha256_file(config.system_prompt),
            "frozen_evaluation": str(config.frozen_evaluation),
            "frozen_evaluation_sha256": _sha256_file(config.frozen_evaluation),
            "config_sha256": _sha256_file(config.config_path),
        },
        "runtime": {
            "llama_cli": str(llama_cli),
            "llama_cli_version": _llama_version(llama_cli),
            "context_size": config.context_size,
            "n_predict": config.n_predict,
            "gpu_layers": config.gpu_layers,
            "temperature": config.temperature,
            "seed": config.seed,
            "reasoning": "off",
        },
        "metrics": _build_metrics(results),
        "cases": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/deployment/llama_cpp_q4km_lora_v1.json"),
    )
    parser.add_argument("--llama-cli", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_deployment_config(args.config)
    llama_cli = resolve_llama_cli(args.llama_cli)
    report = run_deployment_benchmark(config, llama_cli=llama_cli, output_path=args.output)
    print(json.dumps({"deployment_id": report["deployment_id"], "metrics": report["metrics"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
