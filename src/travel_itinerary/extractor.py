"""A transparent baseline extractor used before any fine-tuned model is available."""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Protocol, runtime_checkable

from .contracts import ToolCall, TravelDecision, TravelRequest

KNOWN_CITIES = (
    "北京",
    "成都",
    "重庆",
    "大理",
    "广州",
    "杭州",
    "昆明",
    "南京",
    "厦门",
    "上海",
    "深圳",
    "苏州",
    "三亚",
    "武汉",
    "西安",
    "长沙",
)
CITY_PATTERN = "|".join(sorted(KNOWN_CITIES, key=len, reverse=True))
CHINESE_NUMBERS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
THEME_ALIASES = {
    "亲子": "亲子",
    "美食": "美食",
    "人文": "人文",
    "历史": "人文",
    "自然": "自然",
    "海边": "海边",
    "海岛": "海边",
    "摄影": "摄影",
    "徒步": "徒步",
    "博物馆": "人文",
    "古镇": "人文",
    "休闲": "休闲",
    "温泉": "休闲",
}
HOTEL_PATTERNS = (
    ("市中心", "市中心"),
    ("靠近地铁", "靠近地铁"),
    ("经济型", "经济型"),
    ("性价比", "经济型"),
    ("高档", "高档"),
    ("亲子酒店", "亲子友好"),
)
FIELD_LABELS = {
    "departure_city": "出发城市",
    "destination": "目的地",
    "start_date": "出发日期",
    "days": "旅行天数",
    "traveler_count": "出行人数",
}


@runtime_checkable
class TravelExtractor(Protocol):
    """Shared language-to-decision contract for every inference profile."""

    def decide(self, message: str, today: date) -> TravelDecision:
        """Return a validated decision without executing catalog or write tools."""


class RuleBasedTravelExtractor:
    """A deterministic baseline, deliberately not named or presented as a trained model."""

    def extract(self, message: str, today: date) -> TravelRequest:
        text = self._normalize(message)
        return TravelRequest(
            departure_city=self._parse_departure_city(text),
            destination=self._parse_destination(text),
            start_date=self._parse_start_date(text, today),
            days=self._parse_days(text),
            traveler_count=self._parse_traveler_count(text),
            budget_cny=self._parse_budget(text),
            themes=self._parse_themes(text),
            hotel_preference=self._parse_hotel_preference(text),
        )

    def decide(self, message: str, today: date) -> TravelDecision:
        request = self.extract(message, today)
        missing = request.missing_required_fields
        if missing:
            labels = "、".join(FIELD_LABELS[field] for field in missing)
            return TravelDecision(
                action="clarify",
                request=request,
                missing_fields=missing,
                message=f"为了生成旅行方案，请补充：{labels}。",
            )
        return TravelDecision(
            action="tool_call",
            request=request,
            message="信息已齐全，正在查询本地演示目录中的旅行选项。",
            tool_call=ToolCall(name="search_trip_options", arguments=request.to_tool_arguments()),
        )

    @staticmethod
    def _normalize(message: str) -> str:
        return re.sub(r"\s+", "", message.strip())

    @staticmethod
    def _parse_departure_city(text: str) -> str | None:
        patterns = (
            rf"从(?P<city>{CITY_PATTERN})(?:出发|去|到|前往)",
            rf"(?:出发地|出发城市)(?:是|为)?(?P<city>{CITY_PATTERN})",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group("city")
        return None

    @staticmethod
    def _parse_destination(text: str) -> str | None:
        patterns = (
            rf"(?:去|到|前往)(?P<city>{CITY_PATTERN})(?:旅游|旅行|玩|出差|$|，|。|,|,)",
            rf"(?:目的地)(?:是|为)?(?P<city>{CITY_PATTERN})",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group("city")
        return None

    @staticmethod
    def _parse_start_date(text: str, today: date) -> str | None:
        if "后天" in text:
            return (today + timedelta(days=2)).isoformat()
        if "明天" in text:
            return (today + timedelta(days=1)).isoformat()

        full_date = re.search(r"(?P<year>20\d{2})[年\-/](?P<month>\d{1,2})[月\-/](?P<day>\d{1,2})日?", text)
        if full_date:
            return RuleBasedTravelExtractor._format_date(
                int(full_date.group("year")),
                int(full_date.group("month")),
                int(full_date.group("day")),
            )

        month_day = re.search(r"(?P<month>\d{1,2})月(?P<day>\d{1,2})日?", text)
        if month_day:
            month = int(month_day.group("month"))
            day = int(month_day.group("day"))
            candidate = date(today.year, month, day)
            if candidate < today:
                candidate = date(today.year + 1, month, day)
            return candidate.isoformat()
        return None

    @staticmethod
    def _format_date(year: int, month: int, day: int) -> str | None:
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            return None

    @staticmethod
    def _parse_days(text: str) -> int | None:
        # Do not treat the day portion of a date such as "10月1日" as travel duration.
        digits = re.search(r"(?<!\d)(?P<days>\d{1,2})\s*天", text)
        if digits:
            return int(digits.group("days"))
        chinese = re.search(r"(?P<days>[一二两三四五六七八九十])天", text)
        return CHINESE_NUMBERS.get(chinese.group("days")) if chinese else None

    @staticmethod
    def _parse_traveler_count(text: str) -> int | None:
        digits = re.search(r"(?<!\d)(?P<count>\d{1,2})\s*(?:个)?(?:人|位)", text)
        if digits:
            return int(digits.group("count"))
        chinese = re.search(r"(?P<count>[一二两三四五六七八九十])(?:个)?(?:人|位)", text)
        return CHINESE_NUMBERS.get(chinese.group("count")) if chinese else None

    @staticmethod
    def _parse_budget(text: str) -> int | None:
        match = re.search(
            r"(?:总预算|预算|花费)(?:不超过|约|大约|在)?(?P<budget>\d{3,6})(?:元|块)", text
        )
        return int(match.group("budget")) if match else None

    @staticmethod
    def _parse_themes(text: str) -> tuple[str, ...]:
        matches: list[tuple[int, str]] = []
        for keyword, normalised in THEME_ALIASES.items():
            position = text.find(keyword)
            if position >= 0:
                matches.append((position, normalised))
        themes: list[str] = []
        for _, normalised in sorted(matches):
            if normalised not in themes:
                themes.append(normalised)
        return tuple(themes)

    @staticmethod
    def _parse_hotel_preference(text: str) -> str | None:
        for keyword, normalised in HOTEL_PATTERNS:
            if keyword in text:
                return normalised
        return None
