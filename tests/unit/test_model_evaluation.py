from __future__ import annotations

import json

from travel_itinerary.model_evaluation import (
    Generation,
    extract_first_json_object,
    score_generations,
)


def test_extract_first_json_object_handles_thinking_and_fence_text() -> None:
    payload = extract_first_json_object('<think>ignore</think>\n```json\n{"action":"clarify"}\n```')

    assert payload == {"action": "clarify"}


def test_score_generations_reports_contract_and_exact_match() -> None:
    case = {
        "id": "case-1",
        "today": "2026-08-20",
        "message": "从南京去杭州，10月1日出发，2个人玩3天。",
        "expected_action": "tool_call",
        "expected_slots": {
            "departure_city": "南京",
            "destination": "杭州",
            "start_date": "2026-10-01",
            "days": 3,
            "traveler_count": 2,
            "budget_cny": None,
            "themes": [],
            "hotel_preference": None,
        },
        "expected_tool_arguments": {
            "departure_city": "南京",
            "destination": "杭州",
            "start_date": "2026-10-01",
            "days": 3,
            "traveler_count": 2,
            "budget_cny": None,
            "themes": [],
            "hotel_preference": None,
        },
    }
    response = {
        "action": "tool_call",
        "request": case["expected_slots"],
        "message": "信息已齐全。",
        "tool_call": {"name": "search_trip_options", "arguments": case["expected_tool_arguments"]},
    }

    report = score_generations(
        [case], lambda _: Generation(text=json.dumps(response, ensure_ascii=False), latency_ms=10, output_tokens=8)
    )

    assert report["metrics"]["contract_valid_rate"] == 1.0
    assert report["metrics"]["action_exact_match"] == 1.0
    assert report["metrics"]["tool_argument_exact_match"] == 1.0
