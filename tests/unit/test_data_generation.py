from __future__ import annotations

import json
from pathlib import Path

from travel_itinerary.data_generation import generate_project_data
from travel_itinerary.dataset import load_jsonl


def test_generation_is_reproducible_and_isolated(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = root / "configs/data/travel_v1.0.json"
    prompt = root / "configs/training/travel_system_prompt.txt"

    first = generate_project_data(config, tmp_path / "first", prompt)
    second = generate_project_data(config, tmp_path / "second", prompt)

    assert first["counts"] == {"train": 480, "validation": 80, "evaluation": 120}
    assert first["normalised_input_overlap"] == {
        "train_validation": 0,
        "train_evaluation": 0,
        "validation_evaluation": 0,
    }
    assert first["sha256"] == second["sha256"]
    assert first["human_review"]["status"] == "pending"
    assert first["human_review"]["claim_allowed"] is False


def test_generated_outputs_include_training_and_frozen_eval_shapes(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    output_root = tmp_path / "data"
    generate_project_data(
        root / "configs/data/travel_v1.0.json",
        output_root,
        root / "configs/training/travel_system_prompt.txt",
    )

    train = load_jsonl(output_root / "raw/v1.0/train.jsonl")
    evaluation = load_jsonl(output_root / "eval/v1.0/frozen_cases.jsonl")
    review = load_jsonl(output_root / "review/v1.0/pending_human_review.jsonl")
    manifest = json.loads((output_root / "manifests/travel-v1.0.manifest.json").read_text(encoding="utf-8"))

    assert len(train) == 480
    assert len(evaluation) == 120
    assert len(review) == 30
    assert "rejected" in train[0]
    assert "expected_tool_arguments" in next(case for case in evaluation if case["expected_action"] == "tool_call")
    assert manifest["legacy_term_hit_count"] == 0
    assert manifest["rule_baseline_contract_mismatch_count"] == 0
