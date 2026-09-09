"""CLI entry points for the offline demo, dataset builder, and evaluator."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

from .data_generation import audit_project_data, generate_project_data
from .dataset import build_dataset
from .evaluation import evaluate_file, write_report
from .orchestrator import TravelPlanningOrchestrator
from .runtime import SUPPORTED_EXTRACTOR_MODES, ExtractorRuntimeConfig, build_extractor


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _configure_console_encoding() -> None:
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline travel itinerary assistant PoC")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run one full Clarify -> Tool -> Draft Plan loop")
    demo.add_argument("--message", required=True, help="Chinese natural-language travel request")
    demo.add_argument("--today", type=_parse_date, default=datetime.now().astimezone().date())
    demo.add_argument(
        "--extractor-mode",
        choices=SUPPORTED_EXTRACTOR_MODES,
        default=os.getenv("EXTRACTOR_MODE", "rule"),
        help="explicit language extractor profile (default: EXTRACTOR_MODE or rule)",
    )
    demo.add_argument(
        "--training-config",
        type=Path,
        default=project_root() / "configs/training/qwen3_1_7b_travel_sft_v1.json",
    )
    demo.add_argument(
        "--adapter",
        type=Path,
        default=project_root() / "models/adapters/qwen3-1.7b-travel-sft-v1",
    )
    demo.add_argument(
        "--deployment-config",
        type=Path,
        default=project_root() / "configs/deployment/llama_cpp_q4km_lora_v1.json",
    )
    demo.add_argument("--llama-cli", type=Path)

    build = subparsers.add_parser("build-dataset", help="render seed records to SFT/DPO JSONL")
    build.add_argument("--raw", type=Path, default=project_root() / "data/raw/v0.1/seed.jsonl")
    build.add_argument(
        "--prompt", type=Path, default=project_root() / "configs/training/travel_system_prompt.txt"
    )
    build.add_argument("--output-dir", type=Path, default=project_root() / "data/processed/v0.1")

    generate = subparsers.add_parser(
        "generate-data", help="generate, audit, and manifest deterministic synthetic v1.0 splits"
    )
    generate.add_argument(
        "--config", type=Path, default=project_root() / "configs/data/travel_v1.0.json"
    )
    generate.add_argument("--output-root", type=Path, default=project_root() / "data")
    generate.add_argument(
        "--prompt", type=Path, default=project_root() / "configs/training/travel_system_prompt.txt"
    )

    audit = subparsers.add_parser("audit-data", help="verify persisted v1.0 data against its manifest")
    audit.add_argument("--config", type=Path, default=project_root() / "configs/data/travel_v1.0.json")
    audit.add_argument("--output-root", type=Path, default=project_root() / "data")

    evaluate = subparsers.add_parser("evaluate", help="run the frozen offline decision evaluation")
    evaluate.add_argument("--cases", type=Path, default=project_root() / "data/eval/v1.0/frozen_cases.jsonl")
    evaluate.add_argument(
        "--output", type=Path, default=project_root() / "reports/evaluation/travel_eval_v0.1.json"
    )

    train = subparsers.add_parser("train-sft", help="run the actual local Qwen3 4-bit QLoRA SFT experiment")
    train.add_argument(
        "--config", type=Path, default=project_root() / "configs/training/qwen3_1_7b_travel_sft_v1.json"
    )
    train.add_argument("--validate-only", action="store_true")

    deployment = subparsers.add_parser(
        "benchmark-llama-cpp", help="run the selected frozen smoke suite through local llama.cpp"
    )
    deployment.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs/deployment/llama_cpp_q4km_lora_v1.json",
    )
    deployment.add_argument("--llama-cli", type=Path)
    deployment.add_argument("--output", type=Path)

    quality = subparsers.add_parser(
        "evaluate-llama-cpp",
        help="run all frozen cases through the measured GGUF + LoRA profile",
    )
    quality.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs/deployment/llama_cpp_q4km_lora_v1.json",
    )
    quality.add_argument("--llama-cli", type=Path)
    quality.add_argument(
        "--output",
        type=Path,
        default=project_root() / "reports/deployment/llama_cpp_full_frozen_eval_v1.json",
    )

    performance = subparsers.add_parser(
        "benchmark-llama-runtime",
        help="measure 20+ cold processes and 100+ warm resident requests separately",
    )
    performance.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs/deployment/llama_cpp_q4km_lora_v1.json",
    )
    performance.add_argument("--llama-cli", type=Path)
    performance.add_argument("--llama-server", type=Path)
    performance.add_argument("--cold-runs", type=int, default=20)
    performance.add_argument("--warmup-runs", type=int, default=3)
    performance.add_argument("--warm-runs", type=int, default=100)
    performance.add_argument(
        "--output",
        type=Path,
        default=project_root() / "reports/deployment/llama_cpp_performance_v1.json",
    )

    app_smoke = subparsers.add_parser(
        "smoke-llama-application",
        help="persist one clarify and one model-to-tool-to-plan application smoke",
    )
    app_smoke.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs/deployment/llama_cpp_q4km_lora_v1.json",
    )
    app_smoke.add_argument("--llama-cli", type=Path)
    app_smoke.add_argument(
        "--output",
        type=Path,
        default=project_root() / "reports/application/llama_cpp_application_smoke_v1.json",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    _configure_console_encoding()
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        extractor = build_extractor(
            ExtractorRuntimeConfig(
                mode=args.extractor_mode,
                training_config=args.training_config,
                adapter_path=args.adapter,
                deployment_config=args.deployment_config,
                llama_cli=args.llama_cli,
            )
        )
        orchestrator = TravelPlanningOrchestrator(extractor=extractor)
        response = orchestrator.handle(args.message, args.today)
        print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))
        return
    if args.command == "build-dataset":
        print(json.dumps(build_dataset(args.raw, args.prompt, args.output_dir), ensure_ascii=False, indent=2))
        return
    if args.command == "generate-data":
        print(
            json.dumps(
                generate_project_data(args.config, args.output_root, args.prompt), ensure_ascii=False, indent=2
            )
        )
        return
    if args.command == "audit-data":
        print(json.dumps(audit_project_data(args.config, args.output_root), ensure_ascii=False, indent=2))
        return
    if args.command == "train-sft":
        from .training import load_training_config, run_training

        print(
            json.dumps(
                run_training(load_training_config(args.config), validate_only=args.validate_only),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.command == "benchmark-llama-cpp":
        from .llama_deployment import (
            load_deployment_config,
            resolve_llama_cli,
            run_deployment_benchmark,
        )

        report = run_deployment_benchmark(
            load_deployment_config(args.config),
            llama_cli=resolve_llama_cli(args.llama_cli),
            output_path=args.output,
        )
        print(json.dumps({"deployment_id": report["deployment_id"], "metrics": report["metrics"]}, ensure_ascii=False, indent=2))
        return
    if args.command == "evaluate-llama-cpp":
        from .llama_deployment import load_deployment_config, resolve_llama_cli
        from .llama_quality import run_full_quality_evaluation

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
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.command == "benchmark-llama-runtime":
        from .llama_deployment import load_deployment_config, resolve_llama_cli
        from .llama_performance import resolve_llama_server, run_performance_benchmark

        report = run_performance_benchmark(
            load_deployment_config(args.config),
            llama_cli=resolve_llama_cli(args.llama_cli),
            llama_server=resolve_llama_server(args.llama_server),
            output_path=args.output,
            cold_runs=args.cold_runs,
            warmup_runs=args.warmup_runs,
            warm_runs=args.warm_runs,
        )
        print(
            json.dumps(
                {
                    "cold": report["cold_one_shot"]["metrics"],
                    "warm": report["warm_resident"]["metrics"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.command == "smoke-llama-application":
        from .application_smoke import run_application_smoke
        from .llama_deployment import resolve_llama_cli
        from .runtime import LlamaCppTravelExtractor

        extractor = LlamaCppTravelExtractor(
            args.config,
            resolve_llama_cli(args.llama_cli),
        )
        report = run_application_smoke(extractor)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "passed": report["passed"],
                    "passed_case_count": report["passed_case_count"],
                    "case_count": report["case_count"],
                },
                ensure_ascii=False,
            )
        )
        if not report["passed"]:
            raise SystemExit(1)
        return
    report = evaluate_file(TravelPlanningOrchestrator(), args.cases)
    write_report(args.output, report)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
