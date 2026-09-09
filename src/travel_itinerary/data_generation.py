"""Configuration-driven synthetic travel data with reproducible lineage and quality audits.

The generator is deliberately explicit that the records are synthetic. It creates a review queue
but never marks data as manually reviewed on behalf of a human reviewer.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .contracts import ToolCall, TravelDecision, TravelRequest
from .dataset import build_dataset, load_jsonl, write_jsonl
from .extractor import FIELD_LABELS, RuleBasedTravelExtractor

FORBIDDEN_LEGACY_TERMS = ("按摩", "技师", "预约", "日历写入", "find_technicians")
SPLIT_TEMPLATE_PREFIX = {"train": "tr", "validation": "va", "evaluation": "ev"}


def load_generation_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "version",
        "reference_date",
        "seed",
        "splits",
        "human_review_queue_size",
        "scenario_weights",
        "departure_cities",
        "destinations",
        "themes",
        "hotel_preferences",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"generation config is missing keys: {sorted(missing)}")
    if set(config["splits"]) != {"train", "validation", "evaluation"}:
        raise ValueError("splits must define train, validation, and evaluation")
    if sum(config["scenario_weights"].values()) <= 0:
        raise ValueError("scenario_weights must be positive")
    date.fromisoformat(config["reference_date"])
    return config


def _to_chinese_number(number: int) -> str:
    return {1: "一", 2: "两", 3: "三", 4: "四", 5: "五", 6: "六"}[number]


def _date_phrase(rng: random.Random, today: date) -> tuple[str, str, str]:
    style = rng.choice(("full_date", "month_day", "relative"))
    if style == "relative":
        offset = rng.choice((1, 2))
        return ("明天" if offset == 1 else "后天", (today + timedelta(days=offset)).isoformat(), style)
    candidate = rng.choice(
        (
            date(today.year, 9, 7),
            date(today.year, 9, 19),
            date(today.year, 10, 1),
            date(today.year, 10, 3),
            date(today.year, 10, 18),
            date(today.year, 11, 8),
            date(today.year, 11, 21),
        )
    )
    phrase = (
        f"{candidate.year}年{candidate.month}月{candidate.day}日"
        if style == "full_date"
        else f"{candidate.month}月{candidate.day}日"
    )
    return phrase, candidate.isoformat(), style


def _build_request(rng: random.Random, config: dict[str, Any], scenario: str) -> tuple[TravelRequest, dict[str, str]]:
    today = date.fromisoformat(config["reference_date"])
    departure = rng.choice(config["departure_cities"])
    destinations = [city for city in config["destinations"] if city != departure]
    destination = rng.choice(destinations)
    date_phrase, start_date, date_style = _date_phrase(rng, today)
    days = rng.choice((2, 3, 4, 5))
    travelers = rng.choice((1, 2, 3, 4, 5))
    budget = rng.choice((3200, 4200, 5000, 6800, 8500, 10000))
    themes = tuple(rng.sample(config["themes"], k=rng.choice((1, 2))))
    hotel = rng.choice(config["hotel_preferences"])

    if scenario == "complete_optional_sparse":
        budget = None
        themes = ()
        hotel = None
    missing_by_scenario = {
        "missing_origin": "departure_city",
        "missing_destination": "destination",
        "missing_date": "start_date",
        "missing_days": "days",
        "missing_travelers": "traveler_count",
    }
    missing = missing_by_scenario.get(scenario)
    values: dict[str, Any] = {
        "departure_city": departure,
        "destination": destination,
        "start_date": start_date,
        "days": days,
        "traveler_count": travelers,
        "budget_cny": budget,
        "themes": themes,
        "hotel_preference": hotel,
    }
    if missing:
        values[missing] = None
    metadata = {
        "date_phrase": date_phrase,
        "date_style": "none" if missing == "start_date" else date_style,
        "days_phrase": f"{days}天" if rng.random() < 0.65 else f"{_to_chinese_number(days)}天",
        "traveler_phrase": (
            f"{travelers}个人" if rng.random() < 0.55 else f"{_to_chinese_number(travelers)}个人"
        ),
    }
    return TravelRequest(**values), metadata


def _theme_phrase(themes: tuple[str, ...]) -> str:
    return "和".join(themes)


def _render_message(
    request: TravelRequest,
    metadata: dict[str, str],
    scenario: str,
    split: str,
    template_index: int,
) -> tuple[str, str]:
    prefix = SPLIT_TEMPLATE_PREFIX[split]
    template_id = f"{prefix}-{scenario}-{template_index % 3 + 1}"
    origin = request.departure_city
    destination = request.destination
    date_phrase = metadata["date_phrase"]
    days_phrase = metadata["days_phrase"]
    traveler_phrase = metadata["traveler_phrase"]
    budget_clause = f"，预算{request.budget_cny}元" if request.budget_cny else ""
    theme_clause = f"，喜欢{_theme_phrase(request.themes)}" if request.themes else ""
    hotel_clause = f"，希望住{request.hotel_preference}" if request.hotel_preference else ""

    if scenario == "missing_origin":
        fragments = [f"{date_phrase}去{destination}玩{days_phrase}", traveler_phrase + "出行"]
    elif scenario == "missing_destination":
        fragments = [f"从{origin}出发", f"{date_phrase}旅行{days_phrase}", traveler_phrase + "出行"]
    elif scenario == "missing_date":
        fragments = [f"从{origin}去{destination}玩{days_phrase}", traveler_phrase + "出行"]
    elif scenario == "missing_days":
        fragments = [f"从{origin}去{destination}", f"{date_phrase}出发", traveler_phrase + "出行"]
    elif scenario == "missing_travelers":
        fragments = [f"从{origin}去{destination}玩{days_phrase}", f"{date_phrase}出发"]
    else:
        fragments = [
            f"从{origin}去{destination}玩{days_phrase}",
            f"{date_phrase}出发",
            traveler_phrase + "出行",
        ]

    core = "，".join(fragments)
    if split == "train":
        message = f"我准备{core}{budget_clause}{theme_clause}{hotel_clause}。"
    elif split == "validation":
        message = f"帮我安排一趟旅行：{core}{theme_clause}{budget_clause}{hotel_clause}。"
    else:
        message = f"请规划行程，{core}{hotel_clause}{budget_clause}{theme_clause}。"
    return message, template_id


def _make_expected(request: TravelRequest) -> dict[str, Any]:
    missing = request.missing_required_fields
    if missing:
        labels = "、".join(FIELD_LABELS[field] for field in missing)
        return TravelDecision(
            action="clarify",
            request=request,
            missing_fields=missing,
            message=f"为了生成旅行方案，请补充：{labels}。",
        ).to_dict()
    return TravelDecision(
        action="tool_call",
        request=request,
        message="信息已齐全，正在查询本地演示目录中的旅行选项。",
        tool_call=ToolCall(name="search_trip_options", arguments=request.to_tool_arguments()),
    ).to_dict()


def _make_rejected(expected: dict[str, Any]) -> dict[str, Any]:
    request = expected["request"]
    if expected["action"] == "tool_call":
        return {
            "action": "clarify",
            "request": request,
            "missing_fields": ["destination"],
            "message": "请补充目的地。",
        }
    fabricated = dict(request)
    for key, fallback in (
        ("departure_city", "南京"),
        ("destination", "杭州"),
        ("start_date", "2026-10-01"),
        ("days", 3),
        ("traveler_count", 2),
    ):
        fabricated.setdefault(key, fallback)
        if fabricated[key] is None:
            fabricated[key] = fallback
    return {
        "action": "tool_call",
        "request": fabricated,
        "message": "信息已齐全，正在查询本地演示目录中的旅行选项。",
        "tool_call": {"name": "search_trip_options", "arguments": fabricated},
    }


def _weighted_scenario(rng: random.Random, weights: dict[str, int]) -> str:
    labels = list(weights)
    return rng.choices(labels, weights=[weights[label] for label in labels], k=1)[0]


def generate_split(config: dict[str, Any], split: str) -> list[dict[str, Any]]:
    if split not in config["splits"]:
        raise ValueError(f"unknown split: {split}")
    split_offset = {"train": 11, "validation": 29, "evaluation": 47}[split]
    rng = random.Random(config["seed"] + split_offset)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    target_count = config["splits"][split]
    attempts = 0
    while len(records) < target_count:
        attempts += 1
        if attempts > target_count * 100:
            raise RuntimeError(f"could not generate enough unique {split} examples")
        scenario = _weighted_scenario(rng, config["scenario_weights"])
        request, metadata = _build_request(rng, config, scenario)
        message, template_id = _render_message(request, metadata, scenario, split, len(records))
        normalised = normalize_user_input(message)
        if normalised in seen:
            continue
        seen.add(normalised)
        expected = _make_expected(request)
        record: dict[str, Any] = {
            "id": f"{config['version']}-{split}-{len(records) + 1:04d}",
            "source": "synthetic_template_v1",
            "review_status": "programmatic_passed",
            "today": config["reference_date"],
            "user_input": message,
            "scenario": {"type": scenario, "template_id": template_id, "date_style": metadata["date_style"]},
            "expected": expected,
        }
        if split != "evaluation":
            record["rejected"] = _make_rejected(expected)
        records.append(record)
    return records


def normalize_user_input(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_to_decision(payload: dict[str, Any]) -> TravelDecision:
    raw_request = payload["request"]
    request = TravelRequest(
        departure_city=raw_request["departure_city"],
        destination=raw_request["destination"],
        start_date=raw_request["start_date"],
        days=raw_request["days"],
        traveler_count=raw_request["traveler_count"],
        budget_cny=raw_request["budget_cny"],
        themes=tuple(raw_request["themes"]),
        hotel_preference=raw_request["hotel_preference"],
    )
    if payload["action"] == "clarify":
        return TravelDecision(
            action="clarify",
            request=request,
            message=payload["message"],
            missing_fields=tuple(payload["missing_fields"]),
        )
    raw_tool = payload["tool_call"]
    return TravelDecision(
        action="tool_call",
        request=request,
        message=payload["message"],
        tool_call=ToolCall(name=raw_tool["name"], arguments=raw_tool["arguments"]),
    )


def _coverage(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    scenario_counter: Counter[str] = Counter()
    date_counter: Counter[str] = Counter()
    action_counter: Counter[str] = Counter()
    for record in records:
        scenario_counter[record["scenario"]["type"]] += 1
        date_counter[record["scenario"]["date_style"]] += 1
        action_counter[record["expected"]["action"]] += 1
    return {
        "scenario_type": dict(sorted(scenario_counter.items())),
        "date_style": dict(sorted(date_counter.items())),
        "expected_action": dict(sorted(action_counter.items())),
    }


def audit_splits(
    config: dict[str, Any], train: list[dict[str, Any]], validation: list[dict[str, Any]], evaluation: list[dict[str, Any]]
) -> dict[str, Any]:
    all_splits = {"train": train, "validation": validation, "evaluation": evaluation}
    input_sets = {name: {normalize_user_input(record["user_input"]) for record in records} for name, records in all_splits.items()}
    duplicates = {
        "train_validation": len(input_sets["train"].intersection(input_sets["validation"])),
        "train_evaluation": len(input_sets["train"].intersection(input_sets["evaluation"])),
        "validation_evaluation": len(input_sets["validation"].intersection(input_sets["evaluation"])),
    }
    contract_errors: list[str] = []
    legacy_hits: list[str] = []
    baseline_mismatches: list[str] = []
    extractor = RuleBasedTravelExtractor()
    for records in all_splits.values():
        for record in records:
            try:
                parsed_expected = _payload_to_decision(record["expected"])
                if parsed_expected.to_dict() != record["expected"]:
                    contract_errors.append(record["id"])
            except (KeyError, TypeError, ValueError):
                contract_errors.append(record["id"])
            for term in FORBIDDEN_LEGACY_TERMS:
                if term in json.dumps(record, ensure_ascii=False):
                    legacy_hits.append(f"{record['id']}:{term}")
            baseline = extractor.decide(record["user_input"], date.fromisoformat(record["today"]))
            if baseline.to_dict() != record["expected"]:
                baseline_mismatches.append(record["id"])

    if any(duplicates.values()):
        raise ValueError(f"normalised input leakage detected: {duplicates}")
    if contract_errors:
        raise ValueError(f"invalid expected decision contracts: {contract_errors[:5]}")
    if legacy_hits:
        raise ValueError(f"legacy-domain terms detected: {legacy_hits[:5]}")
    if baseline_mismatches:
        raise ValueError(f"generated examples are not parser-compatible: {baseline_mismatches[:5]}")
    return {
        "version": config["version"],
        "reference_date": config["reference_date"],
        "generator_seed": config["seed"],
        "counts": {name: len(records) for name, records in all_splits.items()},
        "normalised_input_overlap": duplicates,
        "contract_error_count": len(contract_errors),
        "legacy_term_hit_count": len(legacy_hits),
        "rule_baseline_contract_mismatch_count": len(baseline_mismatches),
        "coverage": {name: _coverage(records) for name, records in all_splits.items()},
        "human_review": {
            "status": "pending",
            "claim_allowed": False,
            "message": "A reviewer must record reviewed IDs before manual-review claims are allowed.",
        },
    }


def _write_review_queue(
    output_path: Path, records: list[dict[str, Any]], requested_count: int
) -> int:
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_scenario.setdefault(record["scenario"]["type"], []).append(record)
    queue: list[dict[str, Any]] = []
    offsets: dict[str, int] = {scenario: 0 for scenario in by_scenario}
    while len(queue) < min(requested_count, len(records)):
        made_progress = False
        for scenario in sorted(by_scenario):
            items = by_scenario[scenario]
            index = offsets[scenario]
            if index >= len(items):
                continue
            source = items[index]
            offsets[scenario] += 1
            queue.append(
                {
                    "id": source["id"],
                    "split": source["id"].split("-")[-2],
                    "user_input": source["user_input"],
                    "expected": source["expected"],
                    "scenario": source["scenario"],
                    "review_status": "pending_human_review",
                    "reviewer": None,
                    "reviewed_at": None,
                    "checklist": [
                        "用户表达是否自然且无歧义？",
                        "结构化字段是否只来自用户显式信息或确定性日期归一化？",
                        "该追问或工具调用是否符合字段完整性规则？",
                        "拒绝样本是否代表真实的常见错误？",
                    ],
                }
            )
            made_progress = True
            if len(queue) >= requested_count:
                break
        if not made_progress:
            break
    write_jsonl(output_path, queue)
    return len(queue)


def generate_project_data(config_path: Path, output_root: Path, system_prompt_path: Path) -> dict[str, Any]:
    config = load_generation_config(config_path)
    train = generate_split(config, "train")
    validation = generate_split(config, "validation")
    evaluation = generate_split(config, "evaluation")

    raw_dir = output_root / "raw" / "v1.0"
    eval_dir = output_root / "eval" / "v1.0"
    train_path = raw_dir / "train.jsonl"
    validation_path = raw_dir / "validation.jsonl"
    evaluation_path = eval_dir / "frozen_cases.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)

    eval_records = [
        {
            "id": record["id"],
            "today": record["today"],
            "message": record["user_input"],
            "expected_action": record["expected"]["action"],
            "expected_slots": record["expected"]["request"],
            **(
                {"expected_tool_arguments": record["expected"]["tool_call"]["arguments"]}
                if record["expected"]["action"] == "tool_call"
                else {}
            ),
        }
        for record in evaluation
    ]
    write_jsonl(evaluation_path, eval_records)

    build_dataset(train_path, system_prompt_path, output_root / "processed" / "v1.0" / "train")
    build_dataset(validation_path, system_prompt_path, output_root / "processed" / "v1.0" / "validation")
    audit = audit_splits(config, train, validation, evaluation)
    raw_hashes = {
        "train": _hash_file(train_path),
        "validation": _hash_file(validation_path),
        "evaluation": _hash_file(evaluation_path),
    }
    processed_hashes = {
        "train_sft": _hash_file(output_root / "processed" / "v1.0" / "train" / "sft_train.jsonl"),
        "train_dpo": _hash_file(output_root / "processed" / "v1.0" / "train" / "dpo_train.jsonl"),
        "validation_sft": _hash_file(output_root / "processed" / "v1.0" / "validation" / "sft_train.jsonl"),
    }
    audit["sha256"] = {"raw": raw_hashes, "processed": processed_hashes}
    review_count = _write_review_queue(
        output_root / "review" / "v1.0" / "pending_human_review.jsonl",
        [*train, *validation, *evaluation],
        config["human_review_queue_size"],
    )
    audit["human_review"]["queue_count"] = review_count
    audit["config_sha256"] = _hash_file(config_path)
    manifest_path = output_root / "manifests" / f"{config['version']}.manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def audit_project_data(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = load_generation_config(config_path)
    train = load_jsonl(output_root / "raw" / "v1.0" / "train.jsonl")
    validation = load_jsonl(output_root / "raw" / "v1.0" / "validation.jsonl")
    evaluation = load_jsonl(output_root / "eval" / "v1.0" / "frozen_cases.jsonl")

    # Evaluation records use the public evaluation shape; recover enough to audit source isolation.
    rebuilt_evaluation = []
    for record in evaluation:
        expected = {
            "action": record["expected_action"],
            "request": record["expected_slots"],
            "message": "",
        }
        if record["expected_action"] == "clarify":
            missing = [
                field
                for field, value in record["expected_slots"].items()
                if field in FIELD_LABELS and value is None
            ]
            expected["missing_fields"] = missing
            expected["message"] = "为了生成旅行方案，请补充：" + "、".join(
                FIELD_LABELS[field] for field in missing
            ) + "。"
        else:
            expected["message"] = "信息已齐全，正在查询本地演示目录中的旅行选项。"
            expected["tool_call"] = {
                "name": "search_trip_options",
                "arguments": record["expected_tool_arguments"],
            }
        rebuilt_evaluation.append(
            {
                "id": record["id"],
                "today": record["today"],
                "user_input": record["message"],
                "scenario": {"type": "recovered", "date_style": "recovered"},
                "expected": expected,
            }
        )
    # Full provenance/coverage is preserved in the generated manifest. This method mainly validates
    # persisted split shapes and returns the existing manifest when available.
    manifest_path = output_root / "manifests" / f"{config['version']}.manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("generate-data must run before audit-data")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["counts"] != {
        "train": len(train),
        "validation": len(validation),
        "evaluation": len(rebuilt_evaluation),
    }:
        raise ValueError("persisted split counts do not match manifest")
    return manifest
