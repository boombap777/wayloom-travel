"""The bounded Clarify -> Tool -> Draft Plan orchestration loop."""

from __future__ import annotations

from datetime import date

from .catalog import DemoTripCatalog
from .contracts import TravelDecision, TravelResponse
from .extractor import RuleBasedTravelExtractor, TravelExtractor


class TravelPlanningOrchestrator:
    def __init__(
        self,
        extractor: TravelExtractor | None = None,
        catalog: DemoTripCatalog | None = None,
    ) -> None:
        self.extractor = extractor or RuleBasedTravelExtractor()
        self.catalog = catalog or DemoTripCatalog()

    def decide(self, message: str, today: date) -> TravelDecision:
        return self.extractor.decide(message, today)

    def handle(self, message: str, today: date) -> TravelResponse:
        decision = self.decide(message, today)
        if decision.action == "clarify":
            return TravelResponse(
                action="clarify",
                message=decision.message,
                request=decision.request,
                decision=decision,
            )

        result = self.catalog.search(decision.request)
        plan = self.catalog.build_draft(decision.request, result)
        if plan.selected_option is None:
            message = (
                "本地演示目录中没有满足当前目的地、预算和偏好的选项；"
                "请调整预算或偏好后再查询。"
            )
        else:
            message = "已根据完整需求生成本地演示目录中的旅行草案，请在预订前核验实时信息。"
        return TravelResponse(
            action="final_plan",
            message=message,
            request=decision.request,
            decision=decision,
            plan=plan,
        )
