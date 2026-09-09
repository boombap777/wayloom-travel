"""Application-smoke aggregation without loading a model."""

import json
from pathlib import Path
from types import SimpleNamespace

from travel_itinerary.application_smoke import DEFAULT_CASE_IDS, run_application_smoke
from travel_itinerary.extractor import RuleBasedTravelExtractor


class _FakeExtractor:
    def __init__(self, root: Path) -> None:
        for name in ("base.gguf", "lora.gguf", "prompt.txt", "config.json", "llama-cli"):
            (root / name).write_text(name, encoding="utf-8")
        cases = [
            {
                "id": DEFAULT_CASE_IDS[0],
                "today": "2026-08-20",
                "message": "明天去厦门玩3天，1个人",
                "expected_action": "clarify",
                "expected_slots": {
                    "departure_city": None,
                    "destination": "厦门",
                    "start_date": "2026-08-21",
                    "days": 3,
                    "traveler_count": 1,
                    "budget_cny": None,
                    "themes": [],
                    "hotel_preference": None,
                },
                "expected_tool_arguments": None,
            },
            {
                "id": DEFAULT_CASE_IDS[1],
                "today": "2026-08-20",
                "message": "从重庆去北京玩3天，11月21日出发，一个人出行。",
                "expected_action": "tool_call",
                "expected_slots": {
                    "departure_city": "重庆",
                    "destination": "北京",
                    "start_date": "2026-11-21",
                    "days": 3,
                    "traveler_count": 1,
                    "budget_cny": None,
                    "themes": [],
                    "hotel_preference": None,
                },
                "expected_tool_arguments": {
                    "departure_city": "重庆",
                    "destination": "北京",
                    "start_date": "2026-11-21",
                    "days": 3,
                    "traveler_count": 1,
                    "budget_cny": None,
                    "themes": [],
                    "hotel_preference": None,
                },
            },
        ]
        frozen = root / "frozen.jsonl"
        frozen.write_text(
            "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
            encoding="utf-8",
        )
        self.config = SimpleNamespace(
            frozen_evaluation=frozen,
            base_gguf=root / "base.gguf",
            lora_gguf=root / "lora.gguf",
            system_prompt=root / "prompt.txt",
            config_path=root / "config.json",
        )
        self.llama_cli = root / "llama-cli"
        self.last_run = None
        self._rule = RuleBasedTravelExtractor()

    def decide(self, message, today):
        self.last_run = {"exit_code": 0, "raw_cli_stdout": "fake"}
        return self._rule.decide(message, today)


def test_application_smoke_requires_both_routes_and_catalog_source(tmp_path: Path) -> None:
    report = run_application_smoke(_FakeExtractor(tmp_path))
    assert report["passed"] is True
    assert report["passed_case_count"] == 2
    assert [case["actual_response_action"] for case in report["cases"]] == [
        "clarify",
        "final_plan",
    ]
