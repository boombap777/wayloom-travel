"""Strong, JSON-serialisable contracts shared by extraction, tools, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

REQUIRED_FIELDS: tuple[str, ...] = (
    "departure_city",
    "destination",
    "start_date",
    "days",
    "traveler_count",
)
ALL_SLOT_FIELDS: tuple[str, ...] = REQUIRED_FIELDS + (
    "budget_cny",
    "themes",
    "hotel_preference",
)
DecisionAction = Literal["clarify", "tool_call"]
ResponseAction = Literal["clarify", "final_plan"]


@dataclass(frozen=True, slots=True)
class TravelRequest:
    """Only user-stated or deterministically normalised travel constraints belong here."""

    departure_city: str | None = None
    destination: str | None = None
    start_date: str | None = None
    days: int | None = None
    traveler_count: int | None = None
    budget_cny: int | None = None
    themes: tuple[str, ...] = ()
    hotel_preference: str | None = None

    def __post_init__(self) -> None:
        for name in ("departure_city", "destination", "hotel_preference"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string or null")
        if self.start_date is not None:
            if not isinstance(self.start_date, str):
                raise ValueError("start_date must be a string or null")
            try:
                date.fromisoformat(self.start_date)
            except ValueError as error:
                raise ValueError("start_date must use YYYY-MM-DD") from error
        for name in ("days", "traveler_count", "budget_cny"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer or null")
        if not isinstance(self.themes, tuple) or any(
            not isinstance(theme, str) or not theme.strip() for theme in self.themes
        ):
            raise ValueError("themes must be a tuple of non-empty strings")
        if len(set(self.themes)) != len(self.themes):
            raise ValueError("themes must not contain duplicates")

    @property
    def missing_required_fields(self) -> tuple[str, ...]:
        return tuple(name for name in REQUIRED_FIELDS if getattr(self, name) is None)

    @property
    def is_ready_for_search(self) -> bool:
        return not self.missing_required_fields

    def to_tool_arguments(self) -> dict[str, Any]:
        if not self.is_ready_for_search:
            raise ValueError("cannot build tool arguments before required fields are complete")
        return self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "departure_city": self.departure_city,
            "destination": self.destination,
            "start_date": self.start_date,
            "days": self.days,
            "traveler_count": self.traveler_count,
            "budget_cny": self.budget_cny,
            "themes": list(self.themes),
            "hotel_preference": self.hotel_preference,
        }


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: Literal["search_trip_options"]
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        if self.name != "search_trip_options":
            raise ValueError("unsupported tool name")
        if not isinstance(self.arguments, dict):
            raise TypeError("tool arguments must be an object")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass(frozen=True, slots=True)
class TravelDecision:
    """The model-facing decision before a bounded tool is executed."""

    action: DecisionAction
    request: TravelRequest
    message: str
    missing_fields: tuple[str, ...] = ()
    tool_call: ToolCall | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("decision message must be a non-empty string")
        if self.action == "clarify":
            if not self.missing_fields or self.tool_call is not None:
                raise ValueError("clarify decisions need missing fields and cannot contain a tool call")
            if self.missing_fields != self.request.missing_required_fields:
                raise ValueError("clarify missing_fields must exactly match the incomplete request")
        elif self.action == "tool_call":
            if self.missing_fields or self.tool_call is None or not self.request.is_ready_for_search:
                raise ValueError("tool decisions need a complete request and one tool call")
            if self.tool_call.arguments != self.request.to_tool_arguments():
                raise ValueError("tool arguments must exactly match the validated request")
        else:
            raise ValueError("unsupported decision action")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "action": self.action,
            "request": self.request.to_dict(),
            "message": self.message,
        }
        if self.action == "clarify":
            payload["missing_fields"] = list(self.missing_fields)
        if self.tool_call is not None:
            payload["tool_call"] = self.tool_call.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class CatalogOption:
    destination: str
    title: str
    per_person_budget_cny: int
    themes: tuple[str, ...]
    outline: tuple[str, ...]
    source: str = "demo_catalog_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "destination": self.destination,
            "title": self.title,
            "per_person_budget_cny": self.per_person_budget_cny,
            "themes": list(self.themes),
            "outline": list(self.outline),
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class CatalogResult:
    source: str
    options: tuple[CatalogOption, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "options": [option.to_dict() for option in self.options]}


@dataclass(frozen=True, slots=True)
class DraftPlan:
    catalog_source: str
    selected_option: CatalogOption | None
    estimated_total_cny: int | None
    alternatives: tuple[CatalogOption, ...] = ()
    disclaimer: str = (
        "该方案仅基于本地演示目录生成，不代表实时交通、酒店价格或可订状态。"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_source": self.catalog_source,
            "selected_option": self.selected_option.to_dict() if self.selected_option else None,
            "estimated_total_cny": self.estimated_total_cny,
            "alternatives": [option.to_dict() for option in self.alternatives],
            "disclaimer": self.disclaimer,
        }


@dataclass(frozen=True, slots=True)
class TravelResponse:
    """The user-facing response after the orchestrator handles a decision."""

    action: ResponseAction
    message: str
    request: TravelRequest
    decision: TravelDecision
    plan: DraftPlan | None = None

    def __post_init__(self) -> None:
        if self.action == "clarify" and self.plan is not None:
            raise ValueError("clarification response cannot contain a plan")
        if self.action == "final_plan" and self.plan is None:
            raise ValueError("final plan response needs a plan")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "action": self.action,
            "message": self.message,
            "request": self.request.to_dict(),
            "decision": self.decision.to_dict(),
        }
        if self.plan is not None:
            payload["plan"] = self.plan.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    dataset_version: str
    case_count: int
    metrics: dict[str, float]
    cases: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "case_count": self.case_count,
            "metrics": self.metrics,
            "cases": list(self.cases),
        }
