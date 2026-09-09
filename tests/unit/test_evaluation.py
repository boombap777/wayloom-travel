from __future__ import annotations

from pathlib import Path

from travel_itinerary.evaluation import evaluate_file
from travel_itinerary.orchestrator import TravelPlanningOrchestrator


def test_frozen_evaluation_reports_all_contract_metrics() -> None:
    root = Path(__file__).resolve().parents[2]

    report = evaluate_file(TravelPlanningOrchestrator(), root / "data/eval/v0.1/frozen_cases.jsonl")

    assert report.dataset_version == "travel-eval-v0.1"
    assert report.case_count == 6
    assert set(report.metrics) == {
        "json_valid_rate",
        "action_exact_match",
        "required_slot_micro_f1",
        "tool_argument_exact_match",
    }
    assert all(metric == 1.0 for metric in report.metrics.values())
