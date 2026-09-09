"""Strict and repair-bounded local-model output contracts."""

from copy import deepcopy

import pytest

from travel_itinerary.contracts import ToolCall, TravelDecision, TravelRequest
from travel_itinerary.model_output import (
    ModelOutputError,
    parse_decision_payload,
    repair_missing_tool_arguments,
)


def _request() -> dict:
    return {
        "departure_city": "南京",
        "destination": "杭州",
        "start_date": "2026-10-01",
        "days": 3,
        "traveler_count": 2,
        "budget_cny": 5000,
        "themes": ["人文", "美食"],
        "hotel_preference": "市中心",
    }


def _tool_payload() -> dict:
    request = _request()
    return {
        "action": "tool_call",
        "message": "model text is not authoritative",
        "request": request,
        "tool_call": {"name": "search_trip_options", "arguments": deepcopy(request)},
    }


def test_valid_tool_payload_becomes_canonical_decision() -> None:
    parsed = parse_decision_payload(_tool_payload())
    assert parsed.decision.action == "tool_call"
    assert parsed.decision.message.startswith("信息已齐全")
    assert parsed.decision.tool_call is not None
    assert parsed.decision.tool_call.arguments == parsed.decision.request.to_dict()


def test_valid_clarify_requires_exact_missing_fields() -> None:
    request = _request()
    request["departure_city"] = None
    payload = {
        "action": "clarify",
        "message": "anything",
        "request": request,
        "missing_fields": ["departure_city"],
    }
    parsed = parse_decision_payload(payload)
    assert parsed.decision.missing_fields == ("departure_city",)
    assert "出发城市" in parsed.decision.message


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update({"booking": {"paid": True}}),
        lambda payload: payload["tool_call"].update({"name": "book_hotel"}),
        lambda payload: payload["request"].update({"traveler_count": True}),
        lambda payload: payload["tool_call"].update(
            {"arguments": {**payload["request"], "destination": "北京"}}
        ),
    ],
)
def test_unsafe_or_inconsistent_payload_fails_closed(mutation) -> None:
    payload = _tool_payload()
    mutation(payload)
    with pytest.raises(ModelOutputError):
        parse_decision_payload(payload)


def test_mechanical_repair_only_copies_complete_existing_request() -> None:
    payload = _tool_payload()
    del payload["tool_call"]["arguments"]
    repaired, notes = repair_missing_tool_arguments(payload)
    assert repaired["tool_call"]["arguments"] == payload["request"]
    assert notes == ["rebuilt_tool_arguments_from_complete_model_request"]


def test_mechanical_repair_does_not_fill_an_incomplete_request() -> None:
    payload = _tool_payload()
    payload["request"]["destination"] = None
    del payload["tool_call"]["arguments"]
    repaired, notes = repair_missing_tool_arguments(payload)
    assert "arguments" not in repaired["tool_call"]
    assert notes == []


def test_contract_objects_reject_boolean_integer_and_argument_drift() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        TravelRequest(days=True)

    request = TravelRequest(**{**_request(), "themes": ("人文", "美食")})
    with pytest.raises(ValueError, match="exactly match"):
        TravelDecision(
            action="tool_call",
            request=request,
            message="ready",
            tool_call=ToolCall(
                name="search_trip_options",
                arguments={**request.to_dict(), "destination": "北京"},
            ),
        )
