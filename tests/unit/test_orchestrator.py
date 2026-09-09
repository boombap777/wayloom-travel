from __future__ import annotations

from datetime import date

from travel_itinerary.orchestrator import TravelPlanningOrchestrator


def test_incomplete_request_returns_clarification_without_tool_execution() -> None:
    response = TravelPlanningOrchestrator().handle("我想去厦门玩", date(2026, 8, 20))

    assert response.action == "clarify"
    assert response.plan is None
    assert response.decision.tool_call is None


def test_complete_request_runs_bounded_catalog_and_adds_provenance() -> None:
    response = TravelPlanningOrchestrator().handle(
        "从南京去杭州，10月1日出发，2个人玩3天，预算5000元，喜欢美食。",
        date(2026, 8, 20),
    )

    assert response.action == "final_plan"
    assert response.decision.tool_call is not None
    assert response.plan is not None
    assert response.plan.catalog_source == "demo_catalog_v1"
    assert "不代表实时交通" in response.plan.disclaimer

