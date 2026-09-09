"""Strict frozen-set evaluation for the base and QLoRA adapter models."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ALL_SLOT_FIELDS
from .dataset import load_jsonl
from .model_output import extract_first_json_object, validate_decision_payload
from .prompting import render_generation_prompt
from .training import TrainingConfig, sha256_file


@dataclass(frozen=True, slots=True)
class Generation:
    text: str
    latency_ms: float
    output_tokens: int


def _normalise(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[rank], 3)


def score_generations(
    cases: list[dict[str, Any]],
    generate: Callable[[dict[str, Any]], Generation],
) -> dict[str, Any]:
    json_valid = contract_valid = action_correct = tool_correct = tool_total = 0
    true_positive = false_positive = false_negative = 0
    latencies: list[float] = []
    output_tokens: list[int] = []
    results: list[dict[str, Any]] = []

    for case in cases:
        generation = generate(case)
        parsed_raw = extract_first_json_object(generation.text)
        parsed, contract_error = validate_decision_payload(parsed_raw)
        is_json_valid = parsed_raw is not None
        is_contract_valid = parsed is not None
        json_valid += int(is_json_valid)
        contract_valid += int(is_contract_valid)
        action_match = parsed is not None and parsed["action"] == case["expected_action"]
        action_correct += int(action_match)
        predicted_request = parsed["request"] if parsed is not None else {}
        slot_errors: list[str] = []
        for field in ALL_SLOT_FIELDS:
            expected = _normalise(case["expected_slots"].get(field))
            predicted = _normalise(predicted_request.get(field))
            if expected is None or expected == ():
                if predicted is not None and predicted != ():
                    false_positive += 1
                    slot_errors.append(field)
            elif expected == predicted:
                true_positive += 1
            else:
                false_negative += 1
                if predicted is not None and predicted != ():
                    false_positive += 1
                slot_errors.append(field)
        tool_match: bool | None = None
        if case["expected_action"] == "tool_call":
            tool_total += 1
            expected_arguments = case["expected_tool_arguments"]
            actual_arguments = (
                parsed.get("tool_call", {}).get("arguments") if parsed is not None else None
            )
            tool_match = actual_arguments == expected_arguments
            tool_correct += int(tool_match)
        latencies.append(generation.latency_ms)
        output_tokens.append(generation.output_tokens)
        results.append(
            {
                "id": case["id"],
                "json_valid": is_json_valid,
                "contract_valid": is_contract_valid,
                "contract_error": contract_error,
                "action_match": action_match,
                "tool_arguments_match": tool_match,
                "slot_errors": slot_errors,
                "latency_ms": round(generation.latency_ms, 3),
                "output_tokens": generation.output_tokens,
                "raw_output": generation.text,
            }
        )

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    slot_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    count = len(cases)
    return {
        "metrics": {
            "json_valid_rate": json_valid / count if count else 0.0,
            "contract_valid_rate": contract_valid / count if count else 0.0,
            "action_exact_match": action_correct / count if count else 0.0,
            "slot_micro_f1": slot_f1,
            "tool_argument_exact_match": tool_correct / tool_total if tool_total else 0.0,
            "latency_ms_p50": _percentile(latencies, 0.5),
            "latency_ms_p95": _percentile(latencies, 0.95),
            "latency_ms_mean": round(statistics.fmean(latencies), 3) if latencies else None,
            "output_tokens_mean": round(statistics.fmean(output_tokens), 3) if output_tokens else None,
        },
        "cases": results,
    }


class HuggingFaceGenerator:
    def __init__(self, config: TrainingConfig, adapter_path: Path | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.config = config
        self.adapter_path = adapter_path
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_path, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.quantization_type,
            bnb_4bit_use_double_quant=config.double_quant,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_path,
            local_files_only=True,
            device_map={"": 0},
            quantization_config=quantization,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        if adapter_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
        self.model = model.eval()

    def generate(self, case: dict[str, Any]) -> Generation:
        prompt = render_generation_prompt(
            self.tokenizer,
            self._system_prompt,
            case["today"],
            case["message"],
        )
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.config.generation_max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - started) * 1000
        answer_ids = generated[0, encoded["input_ids"].shape[1] :]
        return Generation(
            text=self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip(),
            latency_ms=latency_ms,
            output_tokens=int(answer_ids.shape[0]),
        )

    @property
    def _system_prompt(self) -> str:
        return (self.config.project_root / "configs/training/travel_system_prompt.txt").read_text(
            encoding="utf-8"
        )


def run_model_evaluation(
    config: TrainingConfig,
    *,
    adapter_path: Path | None,
    output_path: Path,
) -> dict[str, Any]:
    if not config.evaluation_path.is_file():
        raise FileNotFoundError(config.evaluation_path)
    cases = load_jsonl(config.evaluation_path)
    generator = HuggingFaceGenerator(config, adapter_path=adapter_path)
    scored = score_generations(cases, generator.generate)
    report = {
        "report_version": "travel-model-eval-v1",
        "experiment_id": config.experiment_id,
        "model_variant": "adapter" if adapter_path else "base",
        "base_model_id": config.base_model_id,
        "base_model_path": str(config.base_model_path),
        "adapter_path": str(adapter_path) if adapter_path else None,
        "adapter_sha256": sha256_file(adapter_path / "adapter_model.safetensors") if adapter_path else None,
        "evaluation_path": str(config.evaluation_path),
        "evaluation_sha256": sha256_file(config.evaluation_path),
        "case_count": len(cases),
        "generation": {
            "max_new_tokens": config.generation_max_new_tokens,
            "do_sample": config.generation_do_sample,
        },
        **scored,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/training/qwen3_1_7b_travel_sft_v1.json")
    )
    parser.add_argument("--adapter", type=Path, help="Adapter directory; omit to evaluate the base model")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    from .training import load_training_config

    args = _parser().parse_args(argv)
    config = load_training_config(args.config)
    report = run_model_evaluation(config, adapter_path=args.adapter, output_path=args.output)
    print(json.dumps({"model_variant": report["model_variant"], "metrics": report["metrics"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
