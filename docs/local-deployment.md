# 本地 llama.cpp 部署记录

## 目标与边界

本阶段验证的是：已完成的 SFT adapter 能否以 LoRA GGUF 形式加载到 Q4_K_M 基座中，并在一台 Windows + RTX 4060 Laptop GPU 环境中对中文旅行请求生成结构化决策。

最初的 3-case 报告只验证部署 Smoke。项目现已另外保存 120-case GGUF 质量报告，以及 20 次 cold one-shot 和 100 次 post-warm-up 常驻请求的性能报告；三种口径彼此独立，均不是生产服务 SLA。

## 构成

| Artifact | Value |
| --- | --- |
| Base model | Qwen3-1.7B Q4_K_M GGUF |
| Adapter | `qwen3-1.7b-travel-sft-v1-lora-f16.gguf` |
| Runtime | llama.cpp `0.1.2-dev`, build 10545 / commit `a30273376` |
| GPU setting | `-ngl 99` |
| Decoding | `temperature=0`, `seed=20260902`, `reasoning=off`, `n_predict=192` |
| Smoke suite | Frozen cases 0001, 0002, 0004 |

## Contract repair policy

量化模型的一个 tool-call 样例输出了完整 `request` 和合法工具名，但遗漏 `tool_call.arguments`。部署适配层允许且仅允许以下机械修复：

1. action 必须已经是 `tool_call`；
2. tool 必须已经是 `search_trip_options`；
3. `request` 必须通过强类型校验，且全部必填槽位齐全；
4. 仅将该 `request.to_tool_arguments()` 复制到缺失的 `tool_call.arguments`。

它不会补出模型未输出的槽位，也不会改变 action、工具名、日期、人数、目的地或预算。每次修复均写入 `repair_notes`。

## 实测结果

`reports/deployment/llama_cpp_q4km_lora_v1.json` 是该次实测的权威来源：

- 3/3 原始输出均含可解析 JSON；
- 2/3 原始输出通过完整合同；
- 1/3 触发上述机械工具参数修复；
- 修复后 3/3 通过合同和动作匹配，2/2 tool-call 的参数完全匹配冻结标签；
- one-shot wall P50/P95 为 4418/4659 ms；解码速度 P50 为 92.2 token/s。

`reports/deployment/llama_cpp_full_frozen_eval_v1.json` 覆盖全部 120 条冻结样本：

- 120/120 子进程正常退出；
- raw 合同/动作/槽位 F1/工具参数 EM 为 68.33%/60.83%/79.21%/50.00%；
- 18 条只缺工具 arguments 的输出经允许的机械复制后，上述指标为 83.33%/75.83%/90.16%/76.47%；
- 其余失败主要是请求缺少 `traveler_count`/`themes` 等字段，严格解析器不会补值。

`reports/deployment/llama_cpp_performance_v1.json` 分开记录：

- cold：20 个独立进程，20/20 成功，P50/P95 3.54/4.25 s；
- warm：3 次预热后，同一回环地址常驻 server 顺序请求 100 次，100/100 成功，P50/P95 0.98/1.71 s，生成 P50 109.5 token/s；
- 性能成功不等同合同正确，报告同时保留 cold/warm 合同有效率 65%/84%。

## 已知限制

当前 Windows llama.cpp build 的 OpenAI-compatible HTTP 端点已通过 UTF-8 请求并用于常驻性能测量，但该 profile 的结构化质量低于未量化 HF adapter。应用层因此采用严格 fail-closed 校验：不合格输出不会触发本地目录，更不会产生预订、支付或外部写入。常驻测量仅限 `127.0.0.1`，不是生产部署声明。
