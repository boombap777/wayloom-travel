from __future__ import annotations

import json
from pathlib import Path

from travel_itinerary.cli import main


def test_cli_demo_runs_full_loop_and_returns_json(capsys: object) -> None:
    main(
        [
            "demo",
            "--today",
            "2026-08-20",
            "--message",
            "从南京去杭州，10月1日出发，2个人玩3天，预算5000元。",
        ]
    )
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]

    assert payload["action"] == "final_plan"
    assert payload["plan"]["catalog_source"] == "demo_catalog_v1"


def test_cli_evaluate_writes_report(tmp_path: Path, capsys: object) -> None:
    output = tmp_path / "report.json"

    main(["evaluate", "--output", str(output)])
    captured = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]

    assert output.exists()
    assert captured["case_count"] == 120
    assert captured["dataset_version"] == "travel-eval-v1.0"


def test_cli_generates_and_audits_versioned_data(tmp_path: Path, capsys: object) -> None:
    main(["generate-data", "--output-root", str(tmp_path / "data")])
    generated = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    main(["audit-data", "--output-root", str(tmp_path / "data")])
    audited = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]

    assert generated["counts"]["evaluation"] == 120
    assert audited["human_review"]["claim_allowed"] is False
