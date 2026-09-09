from __future__ import annotations

from pathlib import Path

from travel_itinerary.training import load_training_config, preflight


def test_local_sft_config_validates_versioned_inputs(monkeypatch: object, tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    base = tmp_path / "qwen3-1.7b-hf"
    base.mkdir()
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
        (base / filename).write_text("{}", encoding="utf-8")
    monkeypatch.setenv("TRAVEL_BASE_MODEL_PATH", str(base))  # type: ignore[attr-defined]

    config = load_training_config(root / "configs/training/qwen3_1_7b_travel_sft_v1.json")
    result = preflight(config)

    assert result["train_count"] == 480
    assert result["validation_count"] == 80
    assert result["evaluation_sha256"]
    assert config.generation_do_sample is False
