"""Strict, fail-closed parsing for local-model travel decisions."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

from .contracts import ALL_SLOT_FIELDS, REQUIRED_FIELDS, ToolCall, TravelDecision, TravelRequest

_REQUEST_KEYS = frozenset(ALL_SLOT_FIELDS)
_CLARIFY_KEYS = frozenset({"action", "message", "request", "missing_fields"})
_TOOL_KEYS = frozenset({"action", "message", "request", "tool_call"})
_TOOL_CALL_KEYS = frozenset({"name", "arguments"})
_FIELD_LABELS = {
    "departure_city": "出发城市",
    "destination": "目的地",
    "start_date": "出发日期",
    "days": "旅行天数",
    "traveler_count": "出行人数",
}


class ModelOutputError(ValueError):
    """Raised when local-model output cannot safely become a TravelDecision."""


@dataclass(frozen=True, slots=True)
class ParsedDecision:
    decision: TravelDecision
    raw_payload: dict[str, Any]
    repaired_payload: dict[str, Any]
    repair_notes: tuple[str, ...] = ()


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object after an optional model reasoning block."""

    candidate = text.rsplit("</think>", maxsplit=1)[-1] if "</think>" in text else text
    decoder = json.JSONDecoder()
    for index, character in enumerate(candidate):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        return payload if isinstance(payload, dict) else None
    return None


def _require_exact_keys(payload: dict[str, Any], expected: frozenset[str], label: str) -> None:
    actual = frozenset(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ModelOutputError(f"{label}_keys:missing={missing}:unknown={unknown}")


def _parse_request(raw_request: Any) -> TravelRequest:
    if not isinstance(raw_request, dict):
        raise ModelOutputError("request_not_object")
    _require_exact_keys(raw_request, _REQUEST_KEYS, "request")
    raw_themes = raw_request["themes"]
    if not isinstance(raw_themes, list):
        raise ModelOutputError("themes_not_array")
    try:
        return TravelRequest(
            departure_city=raw_request["departure_city"],
            destination=raw_request["destination"],
            start_date=raw_request["start_date"],
            days=raw_request["days"],
            traveler_count=raw_request["traveler_count"],
            budget_cny=raw_request["budget_cny"],
            themes=tuple(raw_themes),
            hotel_preference=raw_request["hotel_preference"],
        )
    except (TypeError, ValueError) as error:
        raise ModelOutputError(f"invalid_request:{error}") from error


def repair_missing_tool_arguments(
    payload: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Copy a complete request into one absent argument object without inference."""

    if payload is None:
        return None, []
    repaired = copy.deepcopy(payload)
    if repaired.get("action") != "tool_call":
        return repaired, []
    raw_tool = repaired.get("tool_call")
    if not isinstance(raw_tool, dict) or raw_tool.get("name") != "search_trip_options":
        return repaired, []
    if "arguments" in raw_tool:
        return repaired, []
    try:
        request = _parse_request(repaired.get("request"))
    except ModelOutputError:
        return repaired, []
    if not request.is_ready_for_search:
        return repaired, []
    raw_tool["arguments"] = request.to_tool_arguments()
    return repaired, ["rebuilt_tool_arguments_from_complete_model_request"]


def _canonical_message(action: str, missing: tuple[str, ...]) -> str:
    if action == "clarify":
        labels = "、".join(_FIELD_LABELS[field] for field in missing)
        return f"为了生成旅行方案，请补充：{labels}。"
    return "信息已齐全，正在查询本地演示目录中的旅行选项。"


def parse_decision_payload(
    payload: dict[str, Any] | None,
    *,
    allow_mechanical_repair: bool = False,
) -> ParsedDecision:
    if payload is None:
        raise ModelOutputError("no_json_object")
    raw_payload = copy.deepcopy(payload)
    repair_notes: list[str] = []
    working = copy.deepcopy(payload)
    if allow_mechanical_repair:
        working, repair_notes = repair_missing_tool_arguments(working)
        if working is None:
            raise ModelOutputError("no_json_object")

    action = working.get("action")
    if action == "clarify":
        _require_exact_keys(working, _CLARIFY_KEYS, "decision")
    elif action == "tool_call":
        _require_exact_keys(working, _TOOL_KEYS, "decision")
    else:
        raise ModelOutputError("unsupported_action")

    if not isinstance(working.get("message"), str):
        raise ModelOutputError("message_not_string")
    request = _parse_request(working["request"])

    try:
        if action == "clarify":
            raw_missing = working["missing_fields"]
            if not isinstance(raw_missing, list) or any(
                not isinstance(field, str) for field in raw_missing
            ):
                raise ModelOutputError("missing_fields_not_string_array")
            missing = tuple(raw_missing)
            if any(field not in REQUIRED_FIELDS for field in missing):
                raise ModelOutputError("unknown_missing_field")
            decision = TravelDecision(
                action="clarify",
                request=request,
                message=_canonical_message(action, missing),
                missing_fields=missing,
            )
        else:
            raw_tool = working["tool_call"]
            if not isinstance(raw_tool, dict):
                raise ModelOutputError("tool_call_not_object")
            _require_exact_keys(raw_tool, _TOOL_CALL_KEYS, "tool_call")
            if raw_tool["name"] != "search_trip_options":
                raise ModelOutputError("unsupported_tool")
            decision = TravelDecision(
                action="tool_call",
                request=request,
                message=_canonical_message(action, ()),
                tool_call=ToolCall(
                    name="search_trip_options",
                    arguments=raw_tool["arguments"],
                ),
            )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, ModelOutputError):
            raise
        raise ModelOutputError(f"invalid_contract:{error}") from error

    return ParsedDecision(
        decision=decision,
        raw_payload=raw_payload,
        repaired_payload=working,
        repair_notes=tuple(repair_notes),
    )


def parse_model_text(text: str, *, allow_mechanical_repair: bool = True) -> ParsedDecision:
    return parse_decision_payload(
        extract_first_json_object(text),
        allow_mechanical_repair=allow_mechanical_repair,
    )


def validate_decision_payload(
    payload: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Compatibility adapter used by the existing model/deployment evaluators."""

    try:
        parsed = parse_decision_payload(payload, allow_mechanical_repair=False)
    except ModelOutputError as error:
        return None, str(error)
    return parsed.decision.to_dict(), None
