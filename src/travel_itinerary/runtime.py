"""Explicit rule, Hugging Face adapter, and llama.cpp extractor profiles."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

from .contracts import TravelDecision
from .extractor import RuleBasedTravelExtractor, TravelExtractor
from .model_output import ModelOutputError, ParsedDecision, parse_model_text

ExtractorMode = Literal["rule", "hf_adapter", "llama_cpp"]
SUPPORTED_EXTRACTOR_MODES = ("rule", "hf_adapter", "llama_cpp")


@dataclass(frozen=True, slots=True)
class ExtractorRuntimeConfig:
    mode: ExtractorMode = "rule"
    training_config: Path | None = None
    adapter_path: Path | None = None
    deployment_config: Path | None = None
    llama_cli: Path | None = None

    @classmethod
    def from_environment(cls, project_root: Path) -> ExtractorRuntimeConfig:
        raw_mode = os.getenv("EXTRACTOR_MODE", "rule").strip().lower()
        if raw_mode not in SUPPORTED_EXTRACTOR_MODES:
            raise ValueError(
                f"EXTRACTOR_MODE must be one of {SUPPORTED_EXTRACTOR_MODES}, got {raw_mode!r}"
            )
        return cls(
            mode=raw_mode,  # type: ignore[arg-type]
            training_config=Path(
                os.getenv(
                    "TRAVEL_TRAINING_CONFIG",
                    project_root / "configs/training/qwen3_1_7b_travel_sft_v1.json",
                )
            ),
            adapter_path=Path(
                os.getenv(
                    "TRAVEL_ADAPTER_PATH",
                    project_root / "models/adapters/qwen3-1.7b-travel-sft-v1",
                )
            ),
            deployment_config=Path(
                os.getenv(
                    "TRAVEL_DEPLOYMENT_CONFIG",
                    project_root / "configs/deployment/llama_cpp_q4km_lora_v1.json",
                )
            ),
            llama_cli=Path(os.environ["TRAVEL_LLAMA_CLI_PATH"])
            if os.getenv("TRAVEL_LLAMA_CLI_PATH")
            else None,
        )


class _ModelTravelExtractor:
    """Parse model text through the one strict, bounded repair policy."""

    profile: str

    def _parse(self, text: str) -> ParsedDecision:
        try:
            return parse_model_text(text, allow_mechanical_repair=True)
        except ModelOutputError as error:
            raise ModelOutputError(f"{self.profile}:{error}") from error


class HuggingFaceAdapterTravelExtractor(_ModelTravelExtractor):
    """Lazy local Transformers/PEFT adapter profile."""

    profile = "hf_adapter"

    def __init__(
        self,
        training_config: Path,
        adapter_path: Path,
        *,
        generator: Any | None = None,
    ) -> None:
        self.training_config = training_config.resolve()
        self.adapter_path = adapter_path.resolve()
        self._generator = generator

    def _load_generator(self) -> Any:
        if self._generator is None:
            from .model_evaluation import HuggingFaceGenerator
            from .training import load_training_config

            config = load_training_config(self.training_config)
            self._generator = HuggingFaceGenerator(config, adapter_path=self.adapter_path)
        return self._generator

    def decide(self, message: str, today: date) -> TravelDecision:
        generation = self._load_generator().generate(
            {"today": today.isoformat(), "message": message}
        )
        return self._parse(generation.text).decision


class LlamaCppTravelExtractor(_ModelTravelExtractor):
    """Measured llama-cli base-GGUF + LoRA-GGUF profile."""

    profile = "llama_cpp"

    def __init__(
        self,
        deployment_config: Path,
        llama_cli: Path | None = None,
        *,
        case_runner: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        from .llama_deployment import load_deployment_config, resolve_llama_cli, run_cli_case

        self.config = load_deployment_config(deployment_config)
        self.llama_cli = resolve_llama_cli(llama_cli)
        self._case_runner = case_runner or run_cli_case
        self.last_run: dict[str, Any] | None = None

    def decide(self, message: str, today: date) -> TravelDecision:
        result = self._case_runner(
            self.config,
            self.llama_cli,
            {
                "id": "application-runtime",
                "today": today.isoformat(),
                "message": message,
                "expected_action": "clarify",
                "expected_tool_arguments": None,
            },
        )
        self.last_run = result
        if result.get("exit_code") != 0:
            raise ModelOutputError(
                f"llama_cpp:process_exit={result.get('exit_code')}:"
                f"{str(result.get('raw_cli_stderr', ''))[:200]}"
            )
        return self._parse(str(result.get("raw_cli_stdout", ""))).decision


def build_extractor(config: ExtractorRuntimeConfig) -> TravelExtractor:
    if config.mode == "rule":
        return RuleBasedTravelExtractor()
    if config.mode == "hf_adapter":
        if config.training_config is None or config.adapter_path is None:
            raise ValueError("hf_adapter mode requires training_config and adapter_path")
        return HuggingFaceAdapterTravelExtractor(config.training_config, config.adapter_path)
    if config.mode == "llama_cpp":
        if config.deployment_config is None:
            raise ValueError("llama_cpp mode requires deployment_config")
        return LlamaCppTravelExtractor(config.deployment_config, config.llama_cli)
    raise ValueError(f"unsupported extractor mode: {config.mode}")
