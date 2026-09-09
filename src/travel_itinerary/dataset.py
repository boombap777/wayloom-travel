"""Versioned seed-data validation and simple SFT/DPO rendering utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
    return records


def validate_seed_record(record: dict[str, Any]) -> None:
    required = {"id", "today", "user_input", "expected"}
    missing = required.difference(record)
    if missing:
        raise ValueError(f"seed record missing keys: {sorted(missing)}")
    expected = record["expected"]
    if expected.get("action") not in {"clarify", "tool_call"}:
        raise ValueError("seed expected.action must be clarify or tool_call")
    if "request" not in expected:
        raise ValueError("seed expected.request is required")
    if expected["action"] == "tool_call" and "tool_call" not in expected:
        raise ValueError("tool_call examples require expected.tool_call")
    if "rejected" in record and not isinstance(record["rejected"], dict):
        raise ValueError("rejected must be an object when supplied")


def render_training_records(
    raw_records: list[dict[str, Any]], system_prompt: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sft_records: list[dict[str, Any]] = []
    dpo_records: list[dict[str, Any]] = []
    for record in raw_records:
        validate_seed_record(record)
        chosen = json.dumps(record["expected"], ensure_ascii=False, sort_keys=True)
        prompt = [
            {"role": "system", "content": system_prompt.strip()},
            {
                "role": "user",
                "content": f"当前日期：{record['today']}\n用户请求：{record['user_input']}",
            },
        ]
        sft_records.append({"id": record["id"], "messages": [*prompt, {"role": "assistant", "content": chosen}]})
        if "rejected" in record:
            dpo_records.append(
                {
                    "id": record["id"],
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": json.dumps(record["rejected"], ensure_ascii=False, sort_keys=True),
                }
            )
    return sft_records, dpo_records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records)
    path.write_text(f"{payload}\n" if payload else "", encoding="utf-8")


def build_dataset(raw_path: Path, system_prompt_path: Path, output_dir: Path) -> dict[str, int]:
    raw_records = load_jsonl(raw_path)
    prompt = system_prompt_path.read_text(encoding="utf-8")
    sft_records, dpo_records = render_training_records(raw_records, prompt)
    write_jsonl(output_dir / "sft_train.jsonl", sft_records)
    write_jsonl(output_dir / "dpo_train.jsonl", dpo_records)
    (output_dir / "dataset_info.json").write_text(
        json.dumps(
            {
                "dataset_version": "travel-seed-v0.1",
                "source": str(raw_path).replace("\\", "/"),
                "sft_count": len(sft_records),
                "dpo_count": len(dpo_records),
                "evidence_boundary": "seed data only; not a completed model-training run",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"sft_count": len(sft_records), "dpo_count": len(dpo_records)}

