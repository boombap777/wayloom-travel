"""Small deterministic contracts for the aggregate evidence gate."""

import json
from pathlib import Path

from travel_itinerary.evidence import REQUIRED_TESTS, _read_junit, sha256_file


def test_sha256_file_changes_with_content(tmp_path: Path) -> None:
    path = tmp_path / "evidence.txt"
    path.write_text("one", encoding="utf-8")
    first = sha256_file(path)
    path.write_text("two", encoding="utf-8")
    assert sha256_file(path) != first


def test_read_junit_keeps_counts_and_test_names(tmp_path: Path) -> None:
    path = tmp_path / "junit.xml"
    path.write_text(
        '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase name="contract"/></testsuite></testsuites>',
        encoding="utf-8",
    )
    result = _read_junit(path)
    assert result["tests"] == 1
    assert result["names"] == {"contract"}


def test_coverage_fixture_shape_is_json_serialisable() -> None:
    payload = {"totals": {"percent_covered": 80.0}}
    assert json.loads(json.dumps(payload))["totals"]["percent_covered"] == 80.0


def test_required_test_set_contains_parameterised_fail_closed_contract() -> None:
    assert "test_unsafe_or_inconsistent_payload_fails_closed" in REQUIRED_TESTS
