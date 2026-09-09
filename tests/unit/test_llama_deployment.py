from __future__ import annotations

from travel_itinerary.llama_deployment import parse_cli_perf, repair_missing_tool_arguments
from travel_itinerary.model_evaluation import validate_decision_payload


def _complete_tool_payload() -> dict[str, object]:
    request = {
        "departure_city": "苏州",
        "destination": "西安",
        "start_date": "2026-08-21",
        "days": 5,
        "traveler_count": 5,
        "budget_cny": 8500,
        "themes": ["休闲", "海边"],
        "hotel_preference": "高档",
    }
    return {
        "action": "tool_call",
        "message": "正在查询本地演示目录中的旅行选项。",
        "request": request,
        "tool_call": {"name": "search_trip_options"},
    }


def test_repair_missing_tool_arguments_only_copies_complete_request() -> None:
    repaired, notes = repair_missing_tool_arguments(_complete_tool_payload())

    assert notes == ["rebuilt_tool_arguments_from_complete_model_request"]
    assert repaired is not None
    assert repaired["tool_call"]["arguments"] == repaired["request"]
    decision, error = validate_decision_payload(repaired)
    assert error is None
    assert decision is not None


def test_repair_does_not_invent_missing_required_slot() -> None:
    payload = _complete_tool_payload()
    payload["request"]["traveler_count"] = None

    repaired, notes = repair_missing_tool_arguments(payload)

    assert notes == []
    assert repaired is not None
    assert "arguments" not in repaired["tool_call"]


def test_parse_cli_perf_reads_llama_cpp_summary() -> None:
    metrics = parse_cli_perf("output\n[ Prompt: 1489.5 t/s | Generation: 92.5 t/s ]\n")

    assert metrics == {"prompt_tokens_per_second": 1489.5, "generation_tokens_per_second": 92.5}


def test_parse_cli_perf_returns_none_when_process_has_no_summary() -> None:
    assert parse_cli_perf("no performance record") == {
        "prompt_tokens_per_second": None,
        "generation_tokens_per_second": None,
    }
