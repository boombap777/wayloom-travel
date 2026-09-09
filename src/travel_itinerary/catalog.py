"""Bounded, versioned local travel options; deliberately not a live provider integration."""

from __future__ import annotations

from .contracts import CatalogOption, CatalogResult, DraftPlan, TravelRequest


class DemoTripCatalog:
    source = "demo_catalog_v1"

    def __init__(self) -> None:
        self._options = (
            CatalogOption(
                destination="杭州",
                title="西湖人文与美食轻旅行",
                per_person_budget_cny=1800,
                themes=("人文", "美食", "自然", "休闲"),
                outline=("西湖及周边漫游", "城市人文与本地美食", "自由活动与返程准备"),
            ),
            CatalogOption(
                destination="成都",
                title="成都美食与慢游路线",
                per_person_budget_cny=2100,
                themes=("美食", "人文", "休闲"),
                outline=("城市街区与美食体验", "人文景点与公园慢游", "自由活动与返程准备"),
            ),
            CatalogOption(
                destination="厦门",
                title="厦门海边与城市漫游",
                per_person_budget_cny=2300,
                themes=("海边", "摄影", "美食", "休闲"),
                outline=("滨海步行与城市初探", "岛屿或海边活动", "自由活动与返程准备"),
            ),
            CatalogOption(
                destination="北京",
                title="北京历史文化入门路线",
                per_person_budget_cny=2600,
                themes=("人文", "博物馆", "美食"),
                outline=("历史街区与城市初探", "博物馆或文化景点", "自由活动与返程准备"),
            ),
            CatalogOption(
                destination="西安",
                title="西安古都人文路线",
                per_person_budget_cny=2000,
                themes=("人文", "美食", "摄影"),
                outline=("古城街区与人文景点", "博物馆或历史遗址", "自由活动与返程准备"),
            ),
        )

    def search(self, request: TravelRequest) -> CatalogResult:
        if not request.is_ready_for_search:
            raise ValueError("catalog search requires a complete travel request")

        candidates = [option for option in self._options if option.destination == request.destination]
        if request.budget_cny is not None:
            candidates = [
                option
                for option in candidates
                if option.per_person_budget_cny * request.traveler_count <= request.budget_cny
            ]

        requested_themes = set(request.themes)
        candidates.sort(
            key=lambda option: (
                -len(requested_themes.intersection(option.themes)),
                option.per_person_budget_cny,
                option.title,
            )
        )
        return CatalogResult(source=self.source, options=tuple(candidates))

    def build_draft(self, request: TravelRequest, result: CatalogResult) -> DraftPlan:
        selected = result.options[0] if result.options else None
        total = selected.per_person_budget_cny * request.traveler_count if selected else None
        return DraftPlan(
            catalog_source=result.source,
            selected_option=selected,
            estimated_total_cny=total,
            alternatives=result.options[1:3],
        )

