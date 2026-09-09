"""Verify the actual pre-QLoRA checkpoint by its published tensor shard hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ORIGIN_FILE = Path(__file__).resolve().parents[2] / "configs/models/qwen3_1_7b_origin.json"


def recorded_origin_matches(artifacts: list[dict], origin: dict) -> bool:
    expected = {(row["filename"], row["bytes"], row["sha256"]) for row in origin["weights"]}
    actual = {(row.get("filename"), row.get("bytes"), row.get("sha256")) for row in artifacts}
    return origin.get("model_id") == "Qwen/Qwen3-1.7B" and expected == actual


def verify_directory(directory: Path, origin: dict) -> dict:
    checks = {}
    for row in origin["weights"]:
        path = directory / row["filename"]
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            checks[row["filename"]] = False
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        checks[row["filename"]] = digest.hexdigest() == row["sha256"]
    return {
        "passed": bool(checks) and all(checks.values()),
        "model_id": origin["model_id"],
        "revision": origin["revision"],
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    result = verify_directory(args.model_dir, json.loads(ORIGIN_FILE.read_text(encoding="utf-8")))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
