"""Prepare a separate model Release asset; never add weights to a Git source release."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = "models/adapters/qwen3-1.7b-travel-sft-v1"
LORA = "models/gguf/qwen3-1.7b-travel-sft-v1-lora-f16.gguf"
BASE = "models/gguf/qwen3-1.7b-base-q4-k-m.gguf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(target: Path) -> bool:
    target = target.resolve()
    manifest = json.loads((target / "MODEL_ASSETS.json").read_text(encoding="utf-8"))
    if not manifest.get("files"):
        return False
    for row in manifest["files"]:
        path = (target / row["path"]).resolve()
        if not path.is_relative_to(target) or not path.is_file():
            return False
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            return False
    return True


def export(target: Path, include_base: bool) -> dict:
    target = target.resolve()
    archive = Path(str(target) + ".zip")
    if target.exists() or archive.exists():
        raise ValueError("Output exists; refusing to overwrite")
    if target.is_relative_to(ROOT) or ROOT.is_relative_to(target):
        raise ValueError("Choose a new output directory outside the source repository")
    files = [ADAPTER + "/adapter_model.safetensors", ADAPTER + "/adapter_config.json", LORA]
    if include_base:
        files.append(BASE)
    for relative in files:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(relative)
    target.mkdir(parents=True)
    rows = []
    for relative in files:
        source, destination = ROOT / relative, target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        original = sha256(source)
        if relative.endswith("adapter_config.json"):
            config = json.loads(source.read_text(encoding="utf-8"))
            config["base_model_name_or_path"] = "Qwen/Qwen3-1.7B"
            config["revision"] = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
            destination.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        else:
            shutil.copyfile(source, destination)
            if sha256(destination) != original:
                raise ValueError("Artifact changed during copy")
        rows.append(
            {
                "path": relative,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "original_sha256": original,
                "metadata_normalized": relative.endswith("adapter_config.json"),
            }
        )
    origin = json.loads(
        (ROOT / "configs/models/qwen3_1_7b_origin.json").read_text(encoding="utf-8")
    )
    shutil.copyfile(
        ROOT / "docs/licenses/QWEN-APACHE-2.0.txt", target / "models/LICENSE.QWEN-APACHE-2.0.txt"
    )
    (target / "models/NOTICE.TRAVEL-QWEN.txt").write_text(
        "Qwen3-1.7B: Copyright 2024 Alibaba Cloud; upstream Apache License 2.0.\n"
        "Modified by Liu Tianxiang for synthetic travel-constraint QLoRA (2026).\n"
        "Adapter trained on 480 synthetic examples; GGUF conversion and base Q4_K_M quantization.\n"
        "Adapter config replaces a machine-specific base path with the verified public model ID/revision.\n"
        "Tensor weights are byte-identical to the recorded local artifacts. Not a booking system.\n",
        encoding="utf-8",
    )
    for name in ("models/LICENSE.QWEN-APACHE-2.0.txt", "models/NOTICE.TRAVEL-QWEN.txt"):
        path = target / name
        rows.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    report = {
        "schema_version": "travel-model-assets-v1",
        "license": "Apache-2.0",
        "base_origin": origin,
        "include_base_gguf": include_base,
        "files": rows,
        "scope": "prepared locally; not uploaded; inference/runtime binaries are separate",
    }
    (target / "MODEL_ASSETS.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not verify(target):
        raise ValueError("Export verification failed")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for path in sorted(target.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(target).as_posix())
    return {
        "directory": str(target),
        "zip": str(archive),
        "zip_sha256": sha256(archive),
        "zip_bytes": archive.stat().st_size,
        "verified": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--include-base", action="store_true")
    args = parser.parse_args()
    if bool(args.output) == bool(args.verify):
        parser.error("Choose exactly one of --output or --verify")
    if args.verify:
        passed = verify(args.verify)
        print(json.dumps({"verified": passed}))
        raise SystemExit(0 if passed else 1)
    print(json.dumps(export(args.output, args.include_base), indent=2))


if __name__ == "__main__":
    main()
