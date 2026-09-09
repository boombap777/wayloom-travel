"""Contracts for cold/warm llama.cpp performance evidence."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from travel_itinerary.llama_performance import (
    _performance_metrics,
    build_server_command,
    percentile,
    run_performance_benchmark,
)


def test_percentile_uses_nearest_rank() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0


def test_server_command_is_loopback_only_and_loads_lora() -> None:
    config = SimpleNamespace(
        base_gguf=Path("base.gguf"),
        lora_gguf=Path("adapter.gguf"),
        gpu_layers=99,
        context_size=2048,
    )
    command = build_server_command(config, Path("llama-server"), 18080)
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--lora") + 1] == "adapter.gguf"
    assert "--reasoning" in command


def test_performance_metrics_keep_success_and_contract_separate() -> None:
    metrics = _performance_metrics(
        [
            {
                "success": True,
                "wall_latency_ms": 10.0,
                "generation_tokens_per_second": 90.0,
                "contract_valid_after_repair": False,
            },
            {
                "success": True,
                "wall_latency_ms": 20.0,
                "generation_tokens_per_second": 100.0,
                "contract_valid_after_repair": True,
            },
        ]
    )
    assert metrics["success_count"] == 2
    assert metrics["contract_valid_after_repair_rate"] == 0.5


def test_benchmark_rejects_insufficient_sample_counts() -> None:
    with pytest.raises(ValueError, match="cold_runs>=20"):
        run_performance_benchmark(
            SimpleNamespace(),
            llama_cli=Path("cli"),
            llama_server=Path("server"),
            output_path=Path("unused.json"),
            cold_runs=3,
            warmup_runs=1,
            warm_runs=10,
        )
