"""Current-source evidence manifest and release gate for resume claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .model_origin import recorded_origin_matches

from .contracts import ALL_SLOT_FIELDS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "reports/release/travel_resume_evidence_manifest_v1.json"
DEFAULT_GATE = PROJECT_ROOT / "reports/release/travel_resume_release_gate_v1.json"
DEFAULT_JUNIT = PROJECT_ROOT / "reports/final-junit.xml"
DEFAULT_COVERAGE = PROJECT_ROOT / "reports/final-coverage.json"

SOURCE_ROOTS = ("src", "tests", "configs", "docs")
SOURCE_EXTRAS = (
    ".gitignore",
    "DEV_SPEC.md",
    "README.md",
    "pyproject.toml",
    "requirements-train.txt",
    "uv.lock",
)
EVIDENCE_FILES = (
    "data/manifests/travel-v1.0.manifest.json",
    "data/raw/v1.0/train.jsonl",
    "data/raw/v1.0/validation.jsonl",
    "data/eval/v1.0/frozen_cases.jsonl",
    "data/processed/v1.0/train/sft_train.jsonl",
    "data/processed/v1.0/train/dpo_train.jsonl",
    "data/processed/v1.0/validation/sft_train.jsonl",
    "models/adapters/qwen3-1.7b-travel-sft-v1/adapter_config.json",
    "models/adapters/qwen3-1.7b-travel-sft-v1/adapter_model.safetensors",
    "models/gguf/qwen3-1.7b-base-q4-k-m.gguf",
    "models/gguf/qwen3-1.7b-travel-sft-v1-lora-f16.gguf",
    "reports/training/qwen3-1.7b-travel-sft-v1.run_manifest.json",
    "reports/model_eval/qwen3-1.7b-base__travel-eval-v1.0.json",
    "reports/model_eval/qwen3-1.7b-travel-sft-v1__travel-eval-v1.0.json",
    "reports/deployment/llama_cpp_q4km_lora_v1.json",
    "reports/deployment/llama_cpp_full_frozen_eval_v1.json",
    "reports/deployment/llama_cpp_performance_v1.json",
    "reports/application/llama_cpp_application_smoke_v1.json",
)
REQUIRED_TESTS = frozenset(
    {
        "test_generation_is_reproducible_and_isolated",
        "test_local_sft_config_validates_versioned_inputs",
        "test_full_quality_report_separates_raw_and_repaired_metrics",
        "test_benchmark_rejects_insufficient_sample_counts",
        "test_application_smoke_requires_both_routes_and_catalog_source",
        "test_unsafe_or_inconsistent_payload_fails_closed",
        "test_mechanical_repair_does_not_fill_an_incomplete_request",
        "test_model_profile_reaches_catalog_then_orchestrator_owned_final_plan",
        "test_invalid_model_output_never_invokes_catalog",
        "test_cli_exposes_all_three_explicit_profiles",
    }
)


class EvidenceError(RuntimeError):
    """Raised when persisted evidence is missing or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EvidenceError(f"missing evidence file: {path}")
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise EvidenceError(f"expected JSON object: {path}")
    return payload


def _iter_source_files(project_root: Path) -> Iterable[Path]:
    for root_name in SOURCE_ROOTS:
        root = project_root / root_name
        for path in root.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                yield path
    for relative in SOURCE_EXTRAS:
        path = project_root / relative
        if path.is_file():
            yield path


def build_source_snapshot(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    entries = []
    for path in sorted(set(_iter_source_files(project_root))):
        entries.append(
            {
                "path": path.relative_to(project_root).as_posix(),
                **_file_evidence(path),
            }
        )
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\n")
    git_head = None
    head_path = project_root / ".git/HEAD"
    if head_path.is_file():
        head_text = head_path.read_text(encoding="utf-8").strip()
        if not head_text.startswith("ref:"):
            git_head = head_text
        else:
            ref_path = project_root / ".git" / head_text.removeprefix("ref:").strip()
            if ref_path.is_file():
                git_head = ref_path.read_text(encoding="ascii").strip()
    return {
        "algorithm": "sha256(path + NUL + file_sha256 + LF)",
        "digest": digest.hexdigest(),
        "file_count": len(entries),
        "files": entries,
        "git_repository_present": (project_root / ".git").is_dir(),
        "git_head": git_head,
    }


def validate_project_facts(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    data = _load_json(project_root / "data/manifests/travel-v1.0.manifest.json")
    training = _load_json(
        project_root / "reports/training/qwen3-1.7b-travel-sft-v1.run_manifest.json"
    )
    base = _load_json(
        project_root / "reports/model_eval/qwen3-1.7b-base__travel-eval-v1.0.json"
    )
    adapter = _load_json(
        project_root
        / "reports/model_eval/qwen3-1.7b-travel-sft-v1__travel-eval-v1.0.json"
    )
    smoke = _load_json(project_root / "reports/deployment/llama_cpp_q4km_lora_v1.json")
    full = _load_json(
        project_root / "reports/deployment/llama_cpp_full_frozen_eval_v1.json"
    )
    performance = _load_json(
        project_root / "reports/deployment/llama_cpp_performance_v1.json"
    )
    application = _load_json(
        project_root / "reports/application/llama_cpp_application_smoke_v1.json"
    )

    frozen_sha = sha256_file(project_root / "data/eval/v1.0/frozen_cases.jsonl")
    adapter_sha = sha256_file(
        project_root
        / "models/adapters/qwen3-1.7b-travel-sft-v1/adapter_model.safetensors"
    )
    base_gguf_sha = sha256_file(
        project_root / "models/gguf/qwen3-1.7b-base-q4-k-m.gguf"
    )
    lora_gguf_sha = sha256_file(
        project_root / "models/gguf/qwen3-1.7b-travel-sft-v1-lora-f16.gguf"
    )

    counts = data.get("counts", {})
    overlaps = data.get("normalised_input_overlap", {})
    train_lora = training.get("lora", {})
    train_config = training.get("config", {})
    base_metrics = base.get("metrics", {})
    adapter_metrics = adapter.get("metrics", {})
    smoke_metrics = smoke.get("metrics", {})
    cold = performance.get("cold_one_shot", {}).get("metrics", {})
    warm = performance.get("warm_resident", {})
    warm_metrics = warm.get("metrics", {})

    checks = {
        "eight_slot_contract": len(ALL_SLOT_FIELDS) == 8,
        "dataset_counts_480_80_120": counts == {"train": 480, "validation": 80, "evaluation": 120},
        "cross_split_overlap_zero": all(value == 0 for value in overlaps.values()),
        "dataset_contract_errors_zero": data.get("contract_error_count") == 0,
        "legacy_domain_hits_zero": data.get("legacy_term_hit_count") == 0,
        "training_complete": training.get("status") == "complete",
        "training_qwen3_1_7b": recorded_origin_matches(
            training.get("base_model", {}).get("artifacts", []),
            _load_json(project_root / "configs/models/qwen3_1_7b_origin.json"),
        ),
        "training_4bit_nf4": train_config.get("quantization_bits") == 4 and train_config.get("quantization_type") == "nf4",
        "trainable_parameters_17432576": train_lora.get("trainable_parameters") == 17_432_576,
        "training_gpu_rtx4060_laptop": training.get("runtime", {}).get("gpu") == "NVIDIA GeForce RTX 4060 Laptop GPU",
        "adapter_hash_current": any(
            item.get("filename") == "adapter_model.safetensors" and item.get("sha256") == adapter_sha
            for item in training.get("adapter_artifacts", [])
        ),
        "base_and_adapter_same_120_frozen_cases": (
            base.get("case_count") == adapter.get("case_count") == 120
            and base.get("evaluation_sha256") == adapter.get("evaluation_sha256") == frozen_sha
        ),
        "base_contract_and_action_zero": base_metrics.get("contract_valid_rate") == 0.0 and base_metrics.get("action_exact_match") == 0.0,
        "adapter_schema_and_action_100": adapter_metrics.get("contract_valid_rate") == 1.0 and adapter_metrics.get("action_exact_match") == 1.0,
        "adapter_slot_f1_99_88": round(float(adapter_metrics.get("slot_micro_f1", -1)) * 100, 2) == 99.88,
        "adapter_tool_exact_98_53": round(float(adapter_metrics.get("tool_argument_exact_match", -1)) * 100, 2) == 98.53,
        "smoke_3_of_3_after_repair": smoke_metrics.get("case_count") == 3 and smoke_metrics.get("contract_valid_rate_after_repair") == 1.0 and smoke_metrics.get("action_exact_match_after_repair") == 1.0,
        "smoke_p95_4_66s": round(float(smoke_metrics.get("one_shot_wall_latency_ms_p95", -1)) / 1000, 2) == 4.66,
        "smoke_generation_p50_92_2": smoke_metrics.get("generation_tokens_per_second_p50") == 92.2,
        "application_smoke_2_of_2": application.get("passed") is True and application.get("case_count") == application.get("passed_case_count") == 2,
        "full_gguf_120_cases_recorded": full.get("case_count") == 120 and len(full.get("cases", [])) == 120 and full.get("process_exit_success_rate") == 1.0,
        "full_gguf_raw_repaired_separate": isinstance(full.get("raw_metrics"), dict) and isinstance(full.get("repaired_metrics"), dict) and full.get("mechanical_repair_count") == 18,
        "cold_20_successful": cold.get("request_count", 0) >= 20 and cold.get("success_count") == cold.get("request_count"),
        "warm_100_successful_after_warmup": warm.get("warmup_request_count", 0) >= 1 and warm_metrics.get("request_count", 0) >= 100 and warm_metrics.get("success_count") == warm_metrics.get("request_count"),
        "report_model_and_data_hashes_current": all(
            (
                report.get("artifacts", {}).get("base_gguf_sha256") == base_gguf_sha
                and report.get("artifacts", {}).get("lora_gguf_sha256") == lora_gguf_sha
                and report.get("artifacts", {}).get("frozen_evaluation_sha256") == frozen_sha
            )
            for report in (smoke, full, performance, application)
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "checks": checks,
        "failed_checks": failed,
        "passed": not failed,
        "resume_claims": {
            "dataset": {"train": 480, "validation": 80, "evaluation": 120, "slots": 8},
            "training": {
                "base_model": "Qwen3-1.7B",
                "quantization": "4-bit NF4 QLoRA",
                "trainable_parameters": train_lora.get("trainable_parameters"),
                "gpu": training.get("runtime", {}).get("gpu"),
            },
            "hf_frozen_evaluation": {
                "case_count": 120,
                "base_contract_valid_rate": base_metrics.get("contract_valid_rate"),
                "base_action_exact_match": base_metrics.get("action_exact_match"),
                "adapter_contract_valid_rate": adapter_metrics.get("contract_valid_rate"),
                "adapter_action_exact_match": adapter_metrics.get("action_exact_match"),
                "adapter_slot_micro_f1": adapter_metrics.get("slot_micro_f1"),
                "adapter_tool_argument_exact_match": adapter_metrics.get("tool_argument_exact_match"),
            },
            "llama_cpp_smoke": smoke_metrics,
            "gguf_full_quality": {
                "case_count": full.get("case_count"),
                "raw_metrics": full.get("raw_metrics"),
                "repaired_metrics": full.get("repaired_metrics"),
            },
            "performance": {"cold": cold, "warm": warm_metrics},
        },
    }


def build_manifest(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    facts = validate_project_facts(project_root)
    if not facts["passed"]:
        raise EvidenceError("project fact checks failed: " + ", ".join(facts["failed_checks"]))
    return {
        "schema_version": "travel-resume-evidence-manifest-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "evidence_scope": "synthetic frozen data and local single-machine execution; not production",
        "source_snapshot": build_source_snapshot(project_root),
        "artifacts": {
            relative: _file_evidence(project_root / relative) for relative in EVIDENCE_FILES
        },
        "facts": facts,
    }


def verify_manifest(
    manifest_path: Path = DEFAULT_MANIFEST,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    checks: dict[str, bool] = {}
    snapshot = build_source_snapshot(project_root)
    recorded_snapshot = manifest.get("source_snapshot", {})
    checks["source_snapshot_current"] = (
        snapshot["digest"] == recorded_snapshot.get("digest")
        and snapshot["file_count"] == recorded_snapshot.get("file_count")
    )
    for relative, expected in manifest.get("artifacts", {}).items():
        path = project_root / relative
        checks[f"artifact:{relative}"] = path.is_file() and (
            sha256_file(path) == expected.get("sha256")
            and path.stat().st_size == expected.get("size_bytes")
        )
    facts = validate_project_facts(project_root)
    checks["project_facts_current"] = facts["passed"] and (
        facts["resume_claims"] == manifest.get("facts", {}).get("resume_claims")
    )
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "source_snapshot_sha256": snapshot["digest"],
        "recorded_source_snapshot_sha256": recorded_snapshot.get("digest"),
        "facts": facts,
    }


def _read_junit(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    names = {
        case.attrib.get("name", "")
        for suite in suites
        for case in suite.iter("testcase")
        if all(case.find(tag) is None for tag in ("skipped", "failure", "error"))
    }
    return {
        "tests": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "failures": sum(int(suite.attrib.get("failures", 0)) for suite in suites),
        "errors": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
        "names": names,
    }


def build_release_gate(
    *,
    manifest_path: Path,
    junit_path: Path,
    coverage_path: Path,
) -> dict[str, Any]:
    evidence = verify_manifest(manifest_path)
    junit = _read_junit(junit_path)
    coverage = _load_json(coverage_path)
    missing_tests = sorted(
        required
        for required in REQUIRED_TESTS
        if not any(
            name == required or name.startswith(f"{required}[")
            for name in junit["names"]
        )
    )
    coverage_percent = float(coverage["totals"]["percent_covered"])
    checks = {
        "evidence_manifest_current": evidence["passed"],
        "tests_zero_failures": junit["failures"] == 0 and junit["errors"] == 0,
        "required_tests_present": not missing_tests,
        "test_suite_size": junit["tests"] >= 47,
        "coverage_report_present": coverage_percent > 0,
    }
    fact_checks = evidence["facts"]["checks"]
    claims = {
        "RC1_TASK_AND_DATA": all(
            fact_checks[name]
            for name in (
                "eight_slot_contract",
                "dataset_counts_480_80_120",
                "cross_split_overlap_zero",
                "dataset_contract_errors_zero",
            )
        ),
        "RC2_QLORA_AND_HF_EVALUATION": all(
            fact_checks[name]
            for name in (
                "training_complete",
                "training_qwen3_1_7b",
                "training_4bit_nf4",
                "trainable_parameters_17432576",
                "base_and_adapter_same_120_frozen_cases",
                "base_contract_and_action_zero",
                "adapter_schema_and_action_100",
                "adapter_slot_f1_99_88",
                "adapter_tool_exact_98_53",
            )
        ),
        "RC3_GGUF_DEPLOYMENT_SMOKE": all(
            fact_checks[name]
            for name in (
                "smoke_3_of_3_after_repair",
                "smoke_p95_4_66s",
                "smoke_generation_p50_92_2",
                "report_model_and_data_hashes_current",
            )
        ),
        "RC4_MODEL_APPLICATION_BOUNDARY": (
            fact_checks["application_smoke_2_of_2"]
            and checks["required_tests_present"]
        ),
        "RC5_GGUF_FULL_QUALITY_EVIDENCE": (
            fact_checks["full_gguf_120_cases_recorded"]
            and fact_checks["full_gguf_raw_repaired_separate"]
        ),
        "RC6_COLD_WARM_PERFORMANCE": (
            fact_checks["cold_20_successful"]
            and fact_checks["warm_100_successful_after_warmup"]
        ),
    }
    return {
        "schema_version": "travel-resume-release-gate-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()) and all(claims.values()),
        "resume_ready": all(checks.values()) and all(claims.values()),
        "claims": {
            name: "SATISFIED" if passed else "PARTIAL"
            for name, passed in claims.items()
        },
        "checks": checks,
        "missing_required_tests": missing_tests,
        "junit": {key: value for key, value in junit.items() if key != "names"},
        "coverage_percent": coverage_percent,
        "source_snapshot_sha256": evidence["source_snapshot_sha256"],
        "evidence": evidence,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    verify = commands.add_parser("verify")
    verify.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    verify.add_argument("--junit", type=Path, default=DEFAULT_JUNIT)
    verify.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    verify.add_argument("--output", type=Path, default=DEFAULT_GATE)
    verify.add_argument("--require-ready", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        payload = build_manifest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "source_files": payload["source_snapshot"]["file_count"],
                    "source_sha256": payload["source_snapshot"]["digest"],
                }
            )
        )
        return 0

    gate = build_release_gate(
        manifest_path=args.manifest,
        junit_path=args.junit,
        coverage_path=args.coverage,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"resume_ready": gate["resume_ready"], "claims": gate["claims"]}))
    return 0 if gate["resume_ready"] or not args.require_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
