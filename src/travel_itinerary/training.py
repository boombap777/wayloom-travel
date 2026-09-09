"""Config-validated local QLoRA SFT for the travel requirement decision model.

All heavyweight dependencies are imported only inside `run_training`, allowing the normal project
test suite to run without a CUDA training environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .dataset import load_jsonl


class TrainingError(RuntimeError):
    """Raised if a reproducible local SFT run cannot safely start or finish."""


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    config_path: Path
    project_root: Path
    experiment_id: str
    base_model_path: Path
    base_model_id: str
    train_path: Path
    validation_path: Path
    data_manifest_path: Path
    evaluation_path: Path
    adapter_dir: Path
    run_manifest_name: str
    training_log_name: str
    seed: int
    max_sequence_length: int
    epochs: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    logging_steps: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: str
    quantization_bits: int
    quantization_type: str
    double_quant: bool
    compute_dtype: str
    generation_max_new_tokens: int
    generation_do_sample: bool


def _require_mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingError(f"{field} must be an object")
    return value


def _positive_int(value: object, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TrainingError(f"{field} must be an integer >= {minimum}")
    return value


def _number(value: object, field: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingError(f"{field} must be numeric")
    result = float(value)
    if not minimum <= result <= maximum:
        raise TrainingError(f"{field} must be in [{minimum}, {maximum}]")
    return result


def _resolve(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TrainingError(f"{field} must be a non-empty path")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_training_config(path: Path) -> TrainingConfig:
    resolved_path = path.resolve()
    try:
        payload = _require_mapping(json.loads(resolved_path.read_text(encoding="utf-8")), "config")
        if payload.get("version") != "travel-qlora-sft-v1":
            raise TrainingError("unsupported training config version")
        root = resolved_path.parents[2]
        base = _require_mapping(payload["base_model"], "base_model")
        data = _require_mapping(payload["data"], "data")
        output = _require_mapping(payload["output"], "output")
        training = _require_mapping(payload["training"], "training")
        lora = _require_mapping(payload["lora"], "lora")
        quant = _require_mapping(payload["quantization"], "quantization")
        generation = _require_mapping(payload["generation"], "generation")
        path_env = base["path_env"]
        if not isinstance(path_env, str) or not path_env:
            raise TrainingError("base_model.path_env must name an environment variable")
        base_path_text = os.environ.get(path_env)
        if not base_path_text:
            raise TrainingError(f"set {path_env} to the local Qwen3 base model directory")
        config = TrainingConfig(
            config_path=resolved_path,
            project_root=root,
            experiment_id=str(payload["experiment_id"]),
            base_model_path=Path(base_path_text).resolve(),
            base_model_id=str(base["model_id"]),
            train_path=_resolve(root, data["train_path"], "data.train_path"),
            validation_path=_resolve(root, data["validation_path"], "data.validation_path"),
            data_manifest_path=_resolve(root, data["data_manifest_path"], "data.data_manifest_path"),
            evaluation_path=_resolve(root, data["evaluation_path"], "data.evaluation_path"),
            adapter_dir=_resolve(root, output["adapter_dir"], "output.adapter_dir"),
            run_manifest_name=str(output["run_manifest_name"]),
            training_log_name=str(output["training_log_name"]),
            seed=_positive_int(training["seed"], "training.seed"),
            max_sequence_length=_positive_int(
                training["max_sequence_length"], "training.max_sequence_length", minimum=128
            ),
            epochs=_positive_int(training["epochs"], "training.epochs"),
            per_device_batch_size=_positive_int(
                training["per_device_batch_size"], "training.per_device_batch_size"
            ),
            gradient_accumulation_steps=_positive_int(
                training["gradient_accumulation_steps"], "training.gradient_accumulation_steps"
            ),
            learning_rate=_number(training["learning_rate"], "training.learning_rate", 1e-7, 1e-2),
            warmup_ratio=_number(training["warmup_ratio"], "training.warmup_ratio", 0.0, 0.5),
            weight_decay=_number(training["weight_decay"], "training.weight_decay", 0.0, 1.0),
            max_grad_norm=_number(training["max_grad_norm"], "training.max_grad_norm", 0.01, 10.0),
            logging_steps=_positive_int(training["logging_steps"], "training.logging_steps"),
            lora_rank=_positive_int(lora["rank"], "lora.rank"),
            lora_alpha=_positive_int(lora["alpha"], "lora.alpha"),
            lora_dropout=_number(lora["dropout"], "lora.dropout", 0.0, 0.5),
            lora_target_modules=str(lora["target_modules"]),
            quantization_bits=_positive_int(quant["bits"], "quantization.bits"),
            quantization_type=str(quant["type"]),
            double_quant=bool(quant["double_quant"]),
            compute_dtype=str(quant["compute_dtype"]),
            generation_max_new_tokens=_positive_int(
                generation["max_new_tokens"], "generation.max_new_tokens"
            ),
            generation_do_sample=bool(generation["do_sample"]),
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, TrainingError):
            raise
        raise TrainingError(f"invalid training config: {resolved_path}") from error
    if config.quantization_bits != 4 or config.quantization_type != "nf4":
        raise TrainingError("this profile requires 4-bit NF4 QLoRA")
    if config.compute_dtype != "bfloat16":
        raise TrainingError("this profile requires bfloat16 compute")
    if config.lora_target_modules != "all-linear":
        raise TrainingError("this profile requires target_modules=all-linear")
    if config.generation_do_sample:
        raise TrainingError("frozen evaluation must use deterministic decoding")
    return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def _validate_model_directory(path: Path) -> None:
    expected = {"config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"}
    missing = sorted(name for name in expected if not (path / name).is_file())
    if missing:
        raise TrainingError(f"base model is incomplete: {missing}")


def _validate_sft_records(path: Path) -> list[dict[str, Any]]:
    records = load_jsonl(path)
    if not records:
        raise TrainingError(f"training file is empty: {path}")
    ids: set[str] = set()
    for record in records:
        if not isinstance(record.get("id"), str) or record["id"] in ids:
            raise TrainingError(f"SFT record IDs must be unique in {path}")
        ids.add(record["id"])
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            raise TrainingError(f"SFT record {record['id']} must contain three messages")
        roles = [message.get("role") for message in messages if isinstance(message, dict)]
        if roles != ["system", "user", "assistant"]:
            raise TrainingError(f"SFT record {record['id']} has an invalid role order")
        if any(not isinstance(message.get("content"), str) for message in messages):
            raise TrainingError(f"SFT record {record['id']} contains a non-string message")
    return records


def preflight(config: TrainingConfig) -> dict[str, Any]:
    _validate_model_directory(config.base_model_path)
    train_records = _validate_sft_records(config.train_path)
    validation_records = _validate_sft_records(config.validation_path)
    if not config.data_manifest_path.is_file() or not config.evaluation_path.is_file():
        raise TrainingError("data manifest and frozen evaluation set must exist")
    manifest = json.loads(config.data_manifest_path.read_text(encoding="utf-8"))
    expected_sha = manifest.get("sha256", {}).get("processed", {}).get("train_sft")
    if expected_sha != sha256_file(config.train_path):
        raise TrainingError("training SFT hash does not match the data manifest")
    if manifest.get("human_review", {}).get("claim_allowed") is not False:
        raise TrainingError("data manifest must retain its explicit human-review boundary")
    return {
        "experiment_id": config.experiment_id,
        "base_model_path": str(config.base_model_path),
        "train_count": len(train_records),
        "validation_count": len(validation_records),
        "train_sha256": sha256_file(config.train_path),
        "validation_sha256": sha256_file(config.validation_path),
        "evaluation_sha256": sha256_file(config.evaluation_path),
        "data_manifest_sha256": sha256_file(config.data_manifest_path),
    }


class TokenizedChatDataset:
    def __init__(self, records: Sequence[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.items: list[dict[str, list[int]]] = []
        for record in records:
            messages = record["messages"]
            prefix = tokenizer.apply_chat_template(
                messages[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            complete = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
            )
            prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            input_ids = tokenizer(complete, add_special_tokens=False)["input_ids"]
            if input_ids[: len(prefix_ids)] != prefix_ids:
                raise TrainingError(f"chat-template prefix mismatch: {record['id']}")
            if len(input_ids) > max_length:
                raise TrainingError(
                    f"sample {record['id']} has {len(input_ids)} tokens > max_sequence_length={max_length}"
                )
            labels = [-100] * len(prefix_ids) + input_ids[len(prefix_ids) :]
            if all(token == -100 for token in labels):
                raise TrainingError(f"sample {record['id']} has no supervised assistant tokens")
            self.items.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


class DataCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: Sequence[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        maximum = max(len(feature["input_ids"]) for feature in features)
        batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = maximum - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def _adapter_files(adapter_dir: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(adapter_dir.glob("*")):
        if path.is_file() and path.name not in {"run_manifest.json", "training_log.json"}:
            files.append({"filename": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return files


def _base_files(base_model_path: Path) -> list[dict[str, Any]]:
    return [
        {"filename": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(base_model_path.glob("model-*.safetensors"))
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_training(config: TrainingConfig, *, validate_only: bool = False) -> dict[str, Any]:
    preflight_result = preflight(config)
    if validate_only:
        return {"status": "validated", **preflight_result}
    if config.adapter_dir.exists() and any(config.adapter_dir.iterdir()):
        raise TrainingError(f"adapter output directory is not empty: {config.adapter_dir}")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise TrainingError("CUDA is required for the configured 4-bit QLoRA run")
    if not torch.cuda.is_bf16_supported():
        raise TrainingError("the selected GPU does not support bfloat16")

    set_seed(config.seed)
    random.seed(config.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = TokenizedChatDataset(
        _validate_sft_records(config.train_path), tokenizer, config.max_sequence_length
    )
    validation_dataset = TokenizedChatDataset(
        _validate_sft_records(config.validation_path), tokenizer, config.max_sequence_length
    )
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.quantization_type,
        bnb_4bit_use_double_quant=config.double_quant,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_path,
        local_files_only=True,
        device_map={"": 0},
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
        ),
    )
    trainable_parameters, total_parameters = model.get_nb_trainable_parameters()
    config.adapter_dir.mkdir(parents=True, exist_ok=False)
    arguments = TrainingArguments(
        output_dir=str(config.adapter_dir),
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        optim="paged_adamw_8bit",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        logging_first_step=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        disable_tqdm=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        seed=config.seed,
        data_seed=config.seed,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    training_result = trainer.train()
    final_evaluation = trainer.evaluate()
    trainer.save_model(str(config.adapter_dir))
    tokenizer.save_pretrained(config.adapter_dir)
    adapter_model = config.adapter_dir / "adapter_model.safetensors"
    if not adapter_model.is_file():
        raise TrainingError("training completed without adapter_model.safetensors")
    elapsed_seconds = time.perf_counter() - started
    training_log = {"log_history": trainer.state.log_history, "final_evaluation": final_evaluation}
    _write_json(config.adapter_dir / config.training_log_name, training_log)
    manifest = {
        "manifest_version": "travel-qlora-sft-run-v1",
        "status": "complete",
        "experiment_id": config.experiment_id,
        "config": asdict(config) | {"config_path": str(config.config_path), "project_root": str(config.project_root), "base_model_path": str(config.base_model_path), "train_path": str(config.train_path), "validation_path": str(config.validation_path), "data_manifest_path": str(config.data_manifest_path), "evaluation_path": str(config.evaluation_path), "adapter_dir": str(config.adapter_dir)},
        "config_sha256": sha256_file(config.config_path),
        "data": preflight_result,
        "base_model": {"model_id": config.base_model_id, "artifacts": _base_files(config.base_model_path)},
        "lora": {
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "target_modules": config.lora_target_modules,
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
        },
        "result": {
            "global_step": trainer.state.global_step,
            "training_loss": training_result.training_loss,
            "eval_loss": final_evaluation.get("eval_loss"),
            "elapsed_seconds": round(elapsed_seconds, 3),
        },
        "runtime": {
            "gpu": torch.cuda.get_device_name(0),
            "cuda_runtime": torch.version.cuda,
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "peft": _package_version("peft"),
            "bitsandbytes": _package_version("bitsandbytes"),
            "accelerate": _package_version("accelerate"),
            "peak_cuda_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 3),
        },
        "adapter_artifacts": _adapter_files(config.adapter_dir),
        "dpo_status": "prepared_data_only_not_executed",
    }
    _write_json(config.adapter_dir / config.run_manifest_name, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/qwen3_1_7b_travel_sft_v1.json"),
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_training(load_training_config(args.config), validate_only=args.validate_only)
    except TrainingError as error:
        print(f"travel QLoRA error: {error}", file=os.sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

