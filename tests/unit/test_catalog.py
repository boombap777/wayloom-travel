from __future__ import annotations

from travel_itinerary.catalog import DemoTripCatalog
from travel_itinerary.contracts import TravelRequest


def _request(**overrides: object) -> TravelRequest:
    values: dict[str, object] = {
        "departure_city": "南京",
        "destination": "杭州",
        "start_date": "2026-10-01",
        "days": 3,
        "traveler_count": 2,
    }
    values.update(overrides)
    return TravelRequest(**values)  # type: ignore[arg-type]


def test_catalog_returns_destination_specific_option() -> None:
    result = DemoTripCatalog().search(_request(themes=("美食",)))

    assert result.source == "demo_catalog_v1"
    assert result.options[0].destination == "杭州"


def test_catalog_does_not_claim_an_option_when_budget_cannot_cover_it() -> None:
    result = DemoTripCatalog().search(_request(budget_cny=1000))

    assert result.options == ()

