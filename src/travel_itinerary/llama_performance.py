"""Cold one-shot and warm resident llama.cpp performance benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from .dataset import load_jsonl
from .llama_deployment import (
    LlamaCppDeploymentConfig,
    _llama_version,
    _sha256_file,
    load_deployment_config,
    resolve_llama_cli,
    run_cli_case,
)
from .model_output import ModelOutputError, extract_first_json_object, parse_model_text
from .prompting import build_messages


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return round(ordered[index], 3)


def resolve_llama_server(explicit_path: Path | None = None) -> Path:
    candidate = explicit_path or (
        Path(os.environ["TRAVEL_LLAMA_SERVER_PATH"])
        if os.getenv("TRAVEL_LLAMA_SERVER_PATH")
        else None
    )
    if candidate is None:
        raise RuntimeError("set TRAVEL_LLAMA_SERVER_PATH or pass --llama-server")
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_server_command(
    config: LlamaCppDeploymentConfig,
    llama_server: Path,
    port: int,
) -> list[str]:
    return [
        str(llama_server),
        "-m",
        str(config.base_gguf),
        "--lora",
        str(config.lora_gguf),
        "-ngl",
        str(config.gpu_layers),
        "-c",
        str(config.context_size),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--alias",
        "travel-local",
        "--reasoning",
        "off",
        "--metrics",
    ]


class ResidentLlamaServer:
    """One loopback-only llama-server process with deterministic cleanup."""

    def __init__(
        self,
        config: LlamaCppDeploymentConfig,
        llama_server: Path,
        *,
        log_dir: Path,
        port: int | None = None,
    ) -> None:
        self.config = config
        self.llama_server = llama_server
        self.port = port or _free_local_port()
        self.log_dir = log_dir
        self.process: subprocess.Popen[bytes] | None = None
        self._stdout = None
        self._stderr = None
        self.startup_ms: float | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._stdout = (self.log_dir / "resident.stdout.log").open("wb")
        self._stderr = (self.log_dir / "resident.stderr.log").open("wb")
        started = time.perf_counter()
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        self.process = subprocess.Popen(
            build_server_command(self.config, self.llama_server, self.port),
            cwd=self.config.project_root,
            stdout=self._stdout,
            stderr=self._stderr,
            creationflags=creationflags,
        )
        deadline = time.monotonic() + self.config.timeout_seconds
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited during startup: {self.process.returncode}")
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=2) as response:
                    health = json.loads(response.read().decode("utf-8"))
                if health.get("status") == "ok":
                    self.startup_ms = round((time.perf_counter() - started) * 1000, 3)
                    return
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                time.sleep(0.25)
        raise TimeoutError("llama-server did not become healthy")

    def request(self, *, system_prompt: str, case: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": "travel-local",
            "messages": build_messages(
                system_prompt,
                str(case["today"]),
                str(case["message"]),
            ),
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "max_tokens": self.config.n_predict,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            wall_latency_ms = round((time.perf_counter() - started) * 1000, 3)
            content = str(body["choices"][0]["message"].get("content", ""))
            try:
                parsed = parse_model_text(content, allow_mechanical_repair=True)
                contract_valid = True
                repair_notes = list(parsed.repair_notes)
            except ModelOutputError as error:
                contract_valid = False
                repair_notes = []
                contract_error = str(error)
            else:
                contract_error = None
            timings = body.get("timings", {})
            usage = body.get("usage", {})
            return {
                "id": case["id"],
                "success": True,
                "wall_latency_ms": wall_latency_ms,
                "completion_tokens": usage.get("completion_tokens"),
                "generation_tokens_per_second": timings.get("predicted_per_second"),
                "json_valid": extract_first_json_object(content) is not None,
                "contract_valid_after_repair": contract_valid,
                "contract_error": contract_error,
                "repair_notes": repair_notes,
                "raw_output": content,
            }
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            return {
                "id": case["id"],
                "success": False,
                "wall_latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "error": f"{type(error).__name__}:{error}",
            }

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=20)
        for handle in (self._stdout, self._stderr):
            if handle is not None:
                handle.close()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _performance_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [item for item in results if item.get("success", True)]
    latencies = [float(item["wall_latency_ms"]) for item in successful]
    speeds = [
        float(item["generation_tokens_per_second"])
        for item in successful
        if item.get("generation_tokens_per_second") is not None
    ]
    return {
        "request_count": len(results),
        "success_count": len(successful),
        "wall_latency_ms_p50": percentile(latencies, 0.5),
        "wall_latency_ms_p95": percentile(latencies, 0.95),
        "wall_latency_ms_mean": round(statistics.fmean(latencies), 3) if latencies else None,
        "generation_tokens_per_second_p50": percentile(speeds, 0.5),
        "generation_tokens_per_second_p95": percentile(speeds, 0.95),
        "contract_valid_after_repair_rate": (
            sum(bool(item.get("contract_valid_after_repair")) for item in successful)
            / len(successful)
            if successful
            else 0.0
        ),
    }


def run_performance_benchmark(
    config: LlamaCppDeploymentConfig,
    *,
    llama_cli: Path,
    llama_server: Path,
    output_path: Path,
    cold_runs: int = 20,
    warmup_runs: int = 3,
    warm_runs: int = 100,
) -> dict[str, Any]:
    if cold_runs < 20 or warm_runs < 100 or warmup_runs < 1:
        raise ValueError("requires cold_runs>=20, warm_runs>=100, and warmup_runs>=1")
    cases = load_jsonl(config.frozen_evaluation)
    system_prompt = config.system_prompt.read_text(encoding="utf-8")

    cold_results = []
    for index in range(cold_runs):
        case = cases[index % len(cases)]
        result = run_cli_case(config, llama_cli, case)
        result["success"] = result.get("exit_code") == 0
        result["contract_valid_after_repair"] = result.get("repaired_contract_valid", False)
        cold_results.append(result)
        if index + 1 == cold_runs or (index + 1) % 5 == 0:
            print(f"Cold one-shot benchmark: {index + 1}/{cold_runs}", flush=True)

    log_dir = output_path.parent / "llama_server_runtime_logs"
    with ResidentLlamaServer(config, llama_server, log_dir=log_dir) as server:
        warmup_results = [
            server.request(system_prompt=system_prompt, case=cases[index % len(cases)])
            for index in range(warmup_runs)
        ]
        warm_results = []
        for index in range(warm_runs):
            warm_results.append(
                server.request(
                    system_prompt=system_prompt,
                    case=cases[(index + warmup_runs) % len(cases)],
                )
            )
            if index + 1 == warm_runs or (index + 1) % 10 == 0:
                print(f"Warm resident benchmark: {index + 1}/{warm_runs}", flush=True)
        startup_ms = server.startup_ms

    report = {
        "report_version": "travel-llama-cpp-performance-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "deployment_id": config.deployment_id,
        "evidence_scope": (
            "Cold results are independent llama-cli processes including model load. Warm results "
            "are sequential HTTP requests after declared warm-up against one loopback-only "
            "llama-server. Neither is a concurrent service load test or production SLA."
        ),
        "artifacts": {
            "base_gguf_sha256": _sha256_file(config.base_gguf),
            "lora_gguf_sha256": _sha256_file(config.lora_gguf),
            "system_prompt_sha256": _sha256_file(config.system_prompt),
            "frozen_evaluation_sha256": _sha256_file(config.frozen_evaluation),
            "deployment_config_sha256": _sha256_file(config.config_path),
            "llama_cli": str(llama_cli),
            "llama_cli_sha256": _sha256_file(llama_cli),
            "llama_server": str(llama_server),
            "llama_server_sha256": _sha256_file(llama_server),
        },
        "runtime": {
            "llama_cli_version": _llama_version(llama_cli),
            "llama_server_version": _llama_version(llama_server),
            "context_size": config.context_size,
            "n_predict": config.n_predict,
            "gpu_layers": config.gpu_layers,
            "temperature": config.temperature,
            "seed": config.seed,
            "reasoning": "off",
        },
        "cold_one_shot": {
            "timing_boundary": "process start -> process exit; includes model and LoRA load",
            "metrics": _performance_metrics(cold_results),
            "cases": cold_results,
        },
        "warm_resident": {
            "timing_boundary": "HTTP POST -> complete response body; sequential requests",
            "server_startup_ms": startup_ms,
            "warmup_request_count": warmup_runs,
            "warmup_results": warmup_results,
            "metrics": _performance_metrics(warm_results),
            "cases": warm_results,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/deployment/llama_cpp_q4km_lora_v1.json"),
    )
    parser.add_argument("--llama-cli", type=Path)
    parser.add_argument("--llama-server", type=Path)
    parser.add_argument("--cold-runs", type=int, default=20)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--warm-runs", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/deployment/llama_cpp_performance_v1.json"),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_performance_benchmark(
        load_deployment_config(args.config),
        llama_cli=resolve_llama_cli(args.llama_cli),
        llama_server=resolve_llama_server(args.llama_server),
        output_path=args.output,
        cold_runs=args.cold_runs,
        warmup_runs=args.warmup_runs,
        warm_runs=args.warm_runs,
    )
    print(
        json.dumps(
            {
                "cold": report["cold_one_shot"]["metrics"],
                "warm": report["warm_resident"]["metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
