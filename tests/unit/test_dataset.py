from __future__ import annotations

from pathlib import Path

from travel_itinerary.dataset import build_dataset, load_jsonl


def test_build_dataset_renders_sft_and_dpo_records(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "processed"

    counts = build_dataset(
        root / "data/raw/v0.1/seed.jsonl",
        root / "configs/training/travel_system_prompt.txt",
        output,
    )

    assert counts == {"sft_count": 4, "dpo_count": 3}
    assert len(load_jsonl(output / "sft_train.jsonl")) == 4
    assert len(load_jsonl(output / "dpo_train.jsonl")) == 3
    assert "not a completed model-training run" in (output / "dataset_info.json").read_text(encoding="utf-8")

