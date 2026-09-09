"""Persisted llama.cpp extractor-to-catalog application smoke evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any

from .dataset import load_jsonl
from .llama_deployment import _sha256_file
from .orchestrator import TravelPlanningOrchestrator
from .runtime import LlamaCppTravelExtractor

DEFAULT_CASE_IDS = (
    "travel-v1.0-evaluation-0002",
    "travel-v1.0-evaluation-0004",
)


def run_application_smoke(
    extractor: LlamaCppTravelExtractor,
    *,
    case_ids: tuple[str, ...] = DEFAULT_CASE_IDS,
) -> dict[str, Any]:
    cases = load_jsonl(extractor.config.frozen_evaluation)
    by_id = {case["id"]: case for case in cases}
    missing = sorted(set(case_ids) - set(by_id))
    if missing:
        raise ValueError(f"application smoke cases missing from frozen split: {missing}")

    orchestrator = TravelPlanningOrchestrator(extractor=extractor)
    evidence = []
    for case_id in case_ids:
        case = by_id[case_id]
        response = orchestrator.handle(case["message"], date.fromisoformat(case["today"]))
        expected_response_action = (
            "clarify" if case["expected_action"] == "clarify" else "final_plan"
        )
        request_match = response.request.to_dict() == case["expected_slots"]
        tool_match: bool | None = None
        if case["expected_action"] == "tool_call":
            tool_match = (
                response.decision.tool_call is not None
                and response.decision.tool_call.arguments
                == case["expected_tool_arguments"]
            )
        source_match = (
            response.plan is not None and response.plan.catalog_source == "demo_catalog_v1"
            if response.action == "final_plan"
            else response.plan is None
        )
        passed = (
            response.action == expected_response_action
            and request_match
            and source_match
            and tool_match is not False
        )
        evidence.append(
            {
                "id": case_id,
                "expected_decision_action": case["expected_action"],
                "expected_response_action": expected_response_action,
                "actual_response_action": response.action,
                "request_match": request_match,
                "tool_arguments_match": tool_match,
                "catalog_source_match": source_match,
                "passed": passed,
                "response": response.to_dict(),
                "extractor_run": extractor.last_run,
            }
        )

    return {
        "report_version": "travel-llama-cpp-application-smoke-v1",
        "evidence_scope": (
            "Two selected frozen synthetic cases proving local-model clarify and "
            "tool_call -> demo_catalog_v1 -> final_plan routing; not a full quality metric."
        ),
        "case_ids": list(case_ids),
        "case_count": len(evidence),
        "passed_case_count": sum(item["passed"] for item in evidence),
        "passed": all(item["passed"] for item in evidence),
        "artifacts": {
            "frozen_evaluation_sha256": _sha256_file(extractor.config.frozen_evaluation),
            "base_gguf_sha256": _sha256_file(extractor.config.base_gguf),
            "lora_gguf_sha256": _sha256_file(extractor.config.lora_gguf),
            "system_prompt_sha256": _sha256_file(extractor.config.system_prompt),
            "deployment_config_sha256": _sha256_file(extractor.config.config_path),
            "llama_cli_sha256": _sha256_file(extractor.llama_cli),
        },
        "cases": evidence,
    }


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
        default=Path("reports/application/llama_cpp_application_smoke_v1.json"),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    from .llama_deployment import resolve_llama_cli

    args = _parser().parse_args(argv)
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
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
