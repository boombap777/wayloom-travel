# 本地 QLoRA SFT 实验协议

本项目的实际后训练实验只在以下前提均满足时执行：

1. `data/manifests/travel-v1.0.manifest.json` 的训练 SHA-256 与当前 SFT 文件一致；
2. 通过 `TRAVEL_BASE_MODEL_PATH` 显式指定本地 Qwen3-1.7B 基座目录；
3. 本机 CUDA、bfloat16、Transformers、PEFT、bitsandbytes 与 Accelerate 可用；
4. 目标 adapter 输出目录为空，避免覆盖已有实验产物。

训练脚本保存 adapter、tokenizer、Trainer log 与 `run_manifest.json`。manifest 记录模型分片哈希、训练/验证/评测数据哈希、配置哈希、种子、LoRA 可训练参数、训练/验证 loss、GPU、依赖版本、峰值 CUDA 显存和 adapter 哈希。

评测时，基座模型与 adapter 使用同一份冻结集、相同 system prompt、`do_sample=false` 和相同 `max_new_tokens`。报告保留每条原始输出、JSON/合同解析错误、字段错误、工具参数匹配和时延，不能只保存聚合分数。

当前 DPO JSONL 是为后续对齐实验准备的数据；没有独立 DPO run manifest 和 report 前，禁止写“已进行 DPO 训练”。

GGUF 应用与评测遵循同一份严格解析合同：raw 与修复后指标分开；仅允许从已完整、已通过类型校验的 `request` 复制缺失的 `tool_call.arguments`，禁止补充任何槽位。应用 Smoke、120-case GGUF 质量、cold one-shot 和 warm resident 性能分别写入独立报告，并由 `travel_itinerary.evidence` 绑定源码、模型、数据和报告哈希。
