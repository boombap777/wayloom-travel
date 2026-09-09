"""Full frozen-set quality evaluation for the llama.cpp GGUF + LoRA profile."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .dataset import load_jsonl
from .llama_deployment import (
    LlamaCppDeploymentConfig,
    _llama_version,
    _sha256_file,
    load_deployment_config,
    resolve_llama_cli,
    run_cli_case,
)
from .model_evaluation import Generation, score_generations


def _score_results(
    cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
    *,
    repaired: bool,
) -> dict[str, Any]:
    by_id = {result["id"]: result for result in results}

    def generate(case: dict[str, Any]) -> Generation:
        result = by_id[case["id"]]
        payload_key = "repaired_decision" if repaired else "raw_model_json"
        payload = result.get(payload_key)
        text = (
            json.dumps(payload, ensure_ascii=False)
            if isinstance(payload, dict)
            else str(result.get("raw_cli_stdout", ""))
        )
        return Generation(
            text=text,
            latency_ms=float(result.get("wall_latency_ms", 0.0)),
            output_tokens=int(result.get("completion_tokens", 0)),
        )

    return score_generations(cases, generate)


def run_full_quality_evaluation(
    config: LlamaCppDeploymentConfig,
    *,
    llama_cli: Path,
    output_path: Path,
    case_runner: Callable[..., dict[str, Any]] = run_cli_case,
) -> dict[str, Any]:
    cases = load_jsonl(config.frozen_evaluation)
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        results.append(case_runner(config, llama_cli, case))
        if index == len(cases) or index % 5 == 0:
            print(f"GGUF frozen evaluation: {index}/{len(cases)}", flush=True)

    raw = _score_results(cases, results, repaired=False)
    repaired = _score_results(cases, results, repaired=True)
    raw_cases = {item["id"]: item for item in raw["cases"]}
    repaired_cases = {item["id"]: item for item in repaired["cases"]}
    case_evidence = []
    for result in results:
        case_id = result["id"]
        case_evidence.append(
            {
                "id": case_id,
                "exit_code": result.get("exit_code"),
                "repair_notes": result.get("repair_notes", []),
                "raw": raw_cases[case_id],
                "repaired": repaired_cases[case_id],
                "generation_tokens_per_second": result.get(
                    "generation_tokens_per_second"
                ),
                "raw_model_json": result.get("raw_model_json"),
                "repaired_decision": result.get("repaired_decision"),
                "raw_cli_stdout": result.get("raw_cli_stdout"),
                "raw_cli_stderr": result.get("raw_cli_stderr"),
            }
        )

    report = {
        "report_version": "travel-llama-cpp-frozen-quality-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "deployment_id": config.deployment_id,
        "measurement_scope": (
            "All cases in the versioned 120-case synthetic frozen split, each run as an "
            "independent llama-cli process. Quality is specific to the Q4_K_M base plus "
            "LoRA GGUF and is not inherited from the Transformers adapter report."
        ),
        "repair_policy": (
            "Only a missing tool_call.arguments object is copied from an already complete, "
            "strictly valid request. No field, action, or tool name is inferred."
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
            "deployment_config": str(config.config_path),
            "deployment_config_sha256": _sha256_file(config.config_path),
            "llama_cli": str(llama_cli),
            "llama_cli_sha256": _sha256_file(llama_cli),
        },
        "runtime": {
            "llama_cli_version": _llama_version(llama_cli),
            "context_size": config.context_size,
            "n_predict": config.n_predict,
            "gpu_layers": config.gpu_layers,
            "temperature": config.temperature,
            "seed": config.seed,
            "reasoning": "off",
        },
        "case_count": len(cases),
        "raw_metrics": raw["metrics"],
        "repaired_metrics": repaired["metrics"],
        "mechanical_repair_count": sum(bool(item.get("repair_notes")) for item in results),
        "process_exit_success_rate": (
            sum(item.get("exit_code") == 0 for item in results) / len(results)
            if results
            else 0.0
        ),
        "cases": case_evidence,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/deployment/llama_cpp_q4km_lora_v1.json"),
    )
    parser.add_argument("--llama-cli", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/deployment/llama_cpp_full_frozen_eval_v1.json"),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_full_quality_evaluation(
        load_deployment_config(args.config),
        llama_cli=resolve_llama_cli(args.llama_cli),
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "case_count": report["case_count"],
                "raw_metrics": report["raw_metrics"],
                "repaired_metrics": report["repaired_metrics"],
                "mechanical_repair_count": report["mechanical_repair_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
