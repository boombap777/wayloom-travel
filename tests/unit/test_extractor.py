from __future__ import annotations

from datetime import date

from travel_itinerary.extractor import RuleBasedTravelExtractor


def test_extracts_explicit_chinese_travel_constraints() -> None:
    extractor = RuleBasedTravelExtractor()

    request = extractor.extract(
        "我想从南京去杭州，10月1日出发玩三天，两个人，预算5000元，喜欢人文和美食。",
        date(2026, 8, 20),
    )

    assert request.to_dict() == {
        "departure_city": "南京",
        "destination": "杭州",
        "start_date": "2026-10-01",
        "days": 3,
        "traveler_count": 2,
        "budget_cny": 5000,
        "themes": ["人文", "美食"],
        "hotel_preference": None,
    }


def test_normalises_relative_date_and_chinese_number() -> None:
    extractor = RuleBasedTravelExtractor()

    request = extractor.extract("从广州到西安，后天出发，两个人玩2天，喜欢人文。", date(2026, 8, 20))

    assert request.start_date == "2026-08-22"
    assert request.traveler_count == 2
    assert request.days == 2


def test_never_invents_missing_required_values() -> None:
    extractor = RuleBasedTravelExtractor()

    decision = extractor.decide("我想去厦门玩，喜欢海边和摄影。", date(2026, 8, 20))

    assert decision.action == "clarify"
    assert decision.request.destination == "厦门"
    assert decision.request.departure_city is None
    assert decision.request.start_date is None
    assert decision.tool_call is None

