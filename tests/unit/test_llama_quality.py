"""Offline tests for the full GGUF quality-report aggregation."""

import json
import sys
from pathlib import Path

from travel_itinerary.llama_deployment import LlamaCppDeploymentConfig
from travel_itinerary.llama_quality import run_full_quality_evaluation


def _request(destination: str | None = "杭州") -> dict:
    return {
        "departure_city": "南京",
        "destination": destination,
        "start_date": "2026-10-01",
        "days": 3,
        "traveler_count": 2,
        "budget_cny": 5000,
        "themes": ["人文"],
        "hotel_preference": None,
    }


def _config(tmp_path: Path) -> LlamaCppDeploymentConfig:
    for name in ("base.gguf", "lora.gguf", "prompt.txt", "config.json"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    cases = [
        {
            "id": "clarify",
            "today": "2026-08-20",
            "message": "missing destination",
            "expected_action": "clarify",
            "expected_slots": _request(None),
            "expected_tool_arguments": None,
        },
        {
            "id": "tool",
            "today": "2026-08-20",
            "message": "complete",
            "expected_action": "tool_call",
            "expected_slots": _request(),
            "expected_tool_arguments": _request(),
        },
    ]
    frozen = tmp_path / "frozen.jsonl"
    frozen.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in cases),
        encoding="utf-8",
    )
    return LlamaCppDeploymentConfig(
        config_path=tmp_path / "config.json",
        project_root=tmp_path,
        version="test",
        deployment_id="test-deployment",
        base_gguf=tmp_path / "base.gguf",
        lora_gguf=tmp_path / "lora.gguf",
        system_prompt=tmp_path / "prompt.txt",
        frozen_evaluation=frozen,
        case_ids=("clarify", "tool"),
        context_size=128,
        n_predict=32,
        gpu_layers=0,
        temperature=0.0,
        seed=1,
        timeout_seconds=10,
        report_path=tmp_path / "smoke.json",
    )


def test_full_quality_report_separates_raw_and_repaired_metrics(tmp_path: Path) -> None:
    config = _config(tmp_path)

    def runner(_config, _cli, case):
        request = case["expected_slots"]
        if case["expected_action"] == "clarify":
            payload = {
                "action": "clarify",
                "message": "missing",
                "request": request,
                "missing_fields": ["destination"],
            }
        else:
            payload = {
                "action": "tool_call",
                "message": "ready",
                "request": request,
                "tool_call": {"name": "search_trip_options"},
            }
        from travel_itinerary.model_output import (
            repair_missing_tool_arguments,
            validate_decision_payload,
        )

        raw_decision, raw_error = validate_decision_payload(payload)
        repaired_payload, notes = repair_missing_tool_arguments(payload)
        repaired_decision, repaired_error = validate_decision_payload(repaired_payload)
        return {
            "id": case["id"],
            "exit_code": 0,
            "wall_latency_ms": 10.0,
            "generation_tokens_per_second": 90.0,
            "completion_tokens": 10,
            "raw_model_json": payload,
            "raw_contract_valid": raw_decision is not None,
            "raw_contract_error": raw_error,
            "repaired_decision": repaired_decision,
            "repaired_contract_valid": repaired_decision is not None,
            "repaired_contract_error": repaired_error,
            "repair_notes": notes,
            "raw_cli_stdout": json.dumps(payload, ensure_ascii=False),
            "raw_cli_stderr": "",
        }

    report = run_full_quality_evaluation(
        config,
        llama_cli=Path(sys.executable),
        output_path=tmp_path / "full.json",
        case_runner=runner,
    )
    assert report["case_count"] == 2
    assert report["raw_metrics"]["contract_valid_rate"] == 0.5
    assert report["repaired_metrics"]["contract_valid_rate"] == 1.0
    assert report["mechanical_repair_count"] == 1
    assert len(report["cases"]) == 2
