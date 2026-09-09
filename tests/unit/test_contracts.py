from __future__ import annotations

import pytest

from travel_itinerary.contracts import ToolCall, TravelDecision, TravelRequest


def test_request_identifies_missing_required_fields() -> None:
    request = TravelRequest(destination="杭州")

    assert request.missing_required_fields == (
        "departure_city",
        "start_date",
        "days",
        "traveler_count",
    )
    assert not request.is_ready_for_search


def test_complete_request_builds_json_safe_tool_arguments() -> None:
    request = TravelRequest(
        departure_city="南京",
        destination="杭州",
        start_date="2026-10-01",
        days=3,
        traveler_count=2,
        themes=("人文",),
    )

    assert request.to_tool_arguments()["themes"] == ["人文"]


def test_invalid_request_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="days"):
        TravelRequest(days=0)


def test_tool_decision_requires_a_complete_request() -> None:
    with pytest.raises(ValueError, match="complete request"):
        TravelDecision(
            action="tool_call",
            request=TravelRequest(destination="杭州"),
            message="query",
            tool_call=ToolCall(name="search_trip_options", arguments={}),
        )

