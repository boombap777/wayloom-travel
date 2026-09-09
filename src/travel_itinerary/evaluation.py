"""Offline evaluation for model decisions, not for live itinerary quality claims."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from .contracts import ALL_SLOT_FIELDS, EvaluationReport
from .dataset import load_jsonl
from .orchestrator import TravelPlanningOrchestrator


def _normalise(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


def evaluate(
    orchestrator: TravelPlanningOrchestrator,
    cases: list[dict[str, Any]],
    dataset_version: str = "travel-eval-v0.1",
) -> EvaluationReport:
    action_correct = 0
    json_valid = 0
    tool_total = 0
    tool_correct = 0
    true_positive = false_positive = false_negative = 0
    case_reports: list[dict[str, Any]] = []

    for case in cases:
        decision = orchestrator.decide(case["message"], date.fromisoformat(case["today"]))
        prediction = decision.request.to_dict()
        expected_slots = case["expected_slots"]
        action_match = decision.action == case["expected_action"]
        action_correct += int(action_match)

        try:
            json.loads(json.dumps(decision.to_dict(), ensure_ascii=False))
            is_json_valid = True
            json_valid += 1
        except (TypeError, json.JSONDecodeError):
            is_json_valid = False

        slot_errors: list[str] = []
        for field in ALL_SLOT_FIELDS:
            expected = _normalise(expected_slots.get(field))
            predicted = _normalise(prediction[field])
            if expected is None or expected == ():
                if predicted is not None and predicted != ():
                    false_positive += 1
                    slot_errors.append(field)
            elif predicted == expected:
                true_positive += 1
            else:
                false_negative += 1
                if predicted is not None and predicted != ():
                    false_positive += 1
                slot_errors.append(field)

        tool_match: bool | None = None
        if case["expected_action"] == "tool_call":
            tool_total += 1
            expected_arguments = case["expected_tool_arguments"]
            actual_arguments = decision.tool_call.arguments if decision.tool_call else None
            tool_match = actual_arguments == expected_arguments
            tool_correct += int(tool_match)

        case_reports.append(
            {
                "id": case["id"],
                "action_match": action_match,
                "json_valid": is_json_valid,
                "tool_arguments_match": tool_match,
                "slot_errors": slot_errors,
            }
        )

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    count = len(cases)
    return EvaluationReport(
        dataset_version=dataset_version,
        case_count=count,
        metrics={
            "json_valid_rate": json_valid / count if count else 0.0,
            "action_exact_match": action_correct / count if count else 0.0,
            "required_slot_micro_f1": f1,
            "tool_argument_exact_match": tool_correct / tool_total if tool_total else 0.0,
        },
        cases=tuple(case_reports),
    )


def evaluate_file(orchestrator: TravelPlanningOrchestrator, path: Path) -> EvaluationReport:
    return evaluate(
        orchestrator,
        load_jsonl(path),
        dataset_version=f"travel-eval-{path.parent.name}",
    )


def write_report(path: Path, report: EvaluationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
