"""Shared extractor profiles and model-to-tool-to-plan authority boundaries."""

from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from travel_itinerary.cli import build_parser
from travel_itinerary.model_evaluation import Generation
from travel_itinerary.model_output import ModelOutputError
from travel_itinerary.orchestrator import TravelPlanningOrchestrator
from travel_itinerary.runtime import (
    ExtractorRuntimeConfig,
    HuggingFaceAdapterTravelExtractor,
    LlamaCppTravelExtractor,
    build_extractor,
)

_VALID_TOOL_JSON = """{
  "action": "tool_call",
  "message": "untrusted model message",
  "request": {
    "departure_city": "南京", "destination": "杭州", "start_date": "2026-10-01",
    "days": 3, "traveler_count": 2, "budget_cny": 5000,
    "themes": ["人文", "美食"], "hotel_preference": "市中心"
  },
  "tool_call": {"name": "search_trip_options"}
}"""


class _FakeGenerator:
    def __init__(self, text: str = _VALID_TOOL_JSON) -> None:
        self.text = text

    def generate(self, case: dict) -> Generation:
        return Generation(text=self.text, latency_ms=1.0, output_tokens=10)


def test_rule_profile_is_the_default() -> None:
    extractor = build_extractor(ExtractorRuntimeConfig())
    assert extractor.__class__.__name__ == "RuleBasedTravelExtractor"


def test_hf_adapter_profile_uses_shared_strict_parser_without_loading_model(tmp_path: Path) -> None:
    extractor = HuggingFaceAdapterTravelExtractor(
        tmp_path / "training.json",
        tmp_path / "adapter",
        generator=_FakeGenerator(),
    )
    decision = extractor.decide("ignored", date(2026, 8, 20))
    assert decision.action == "tool_call"
    assert decision.tool_call is not None
    assert decision.tool_call.arguments == decision.request.to_dict()


def test_llama_cpp_profile_runs_through_same_parser(monkeypatch, tmp_path: Path) -> None:
    import travel_itinerary.llama_deployment as deployment

    monkeypatch.setattr(
        deployment,
        "load_deployment_config",
        lambda _: SimpleNamespace(project_root=tmp_path),
    )
    monkeypatch.setattr(deployment, "resolve_llama_cli", lambda _: tmp_path / "llama-cli")
    runner = MagicMock(
        return_value={
            "exit_code": 0,
            "raw_cli_stdout": _VALID_TOOL_JSON,
            "raw_cli_stderr": "",
        }
    )
    extractor = LlamaCppTravelExtractor(
        tmp_path / "deployment.json",
        case_runner=runner,
    )
    decision = extractor.decide("request", date(2026, 8, 20))
    assert decision.action == "tool_call"
    runner.assert_called_once()


def test_model_profile_reaches_catalog_then_orchestrator_owned_final_plan(tmp_path: Path) -> None:
    extractor = HuggingFaceAdapterTravelExtractor(
        tmp_path / "training.json",
        tmp_path / "adapter",
        generator=_FakeGenerator(),
    )
    response = TravelPlanningOrchestrator(extractor=extractor).handle(
        "request",
        date(2026, 8, 20),
    )
    assert response.action == "final_plan"
    assert response.plan is not None
    assert response.plan.catalog_source == "demo_catalog_v1"
    assert "实时" in response.plan.disclaimer


def test_invalid_model_output_never_invokes_catalog(tmp_path: Path) -> None:
    extractor = HuggingFaceAdapterTravelExtractor(
        tmp_path / "training.json",
        tmp_path / "adapter",
        generator=_FakeGenerator('{"action":"book_and_pay"}'),
    )
    catalog = MagicMock()
    orchestrator = TravelPlanningOrchestrator(extractor=extractor, catalog=catalog)
    with pytest.raises(ModelOutputError):
        orchestrator.handle("malicious", date(2026, 8, 20))
    catalog.search.assert_not_called()
    catalog.build_draft.assert_not_called()


def test_cli_exposes_all_three_explicit_profiles() -> None:
    parser = build_parser()
    for mode in ("rule", "hf_adapter", "llama_cpp"):
        args = parser.parse_args(
            ["demo", "--message", "test", "--today", "2026-08-20", "--extractor-mode", mode]
        )
        assert args.extractor_mode == mode
