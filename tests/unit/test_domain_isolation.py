from __future__ import annotations

from pathlib import Path


def test_seed_and_frozen_eval_data_do_not_reuse_appointment_domain_terms() -> None:
    root = Path(__file__).resolve().parents[2]
    corpus = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            root / "data/raw/v0.1/seed.jsonl",
            root / "data/eval/v0.1/frozen_cases.jsonl",
        )
    )

    for forbidden in ("按摩", "技师", "预约", "日历写入", "find_technicians"):
        assert forbidden not in corpus

