# Wayloom Travel · 织途旅行

仓库名与工程目录名：`wayloom-travel`。为保持兼容，内部 Python 分发包 `travel-itinerary-assistant`、
导入路径与现有命令行入口不随此次命名变更重写。原实验报告中的历史名称仍保留。

## 先在浏览器体验

安装 Python 3.12 和 uv，在仓库根目录运行：

```shell
uv sync --frozen
uv run --no-sync travel-itinerary-web
```

打开 http://127.0.0.1:8504/。页面支持需求输入、缺失信息补全、来源明确的草案和 JSON
导出；默认 `rule` 模式不需要模型，也不冒充微调结果。真实 `hf_adapter` / `llama_cpp`
模式与制品准备见 [PUBLICATION.md](PUBLICATION.md)。UI、API 和命令行共享同一规划器。

模型标注更正：原训练配置误填 `Qwen3-1.7B-Base`，实际分片哈希对应固定版本的
`Qwen/Qwen3-1.7B`。原始报告保留，证据和正确下载版本见 [更正记录](docs/model-provenance.md)。

面向中文旅行需求的本地优先技术验证原型（PoC）。系统将自然语言请求转换为强类型旅行约束；信息不全时追问，信息齐全时调用受限的本地演示目录，并生成带数据来源说明的行程草案。

项目与预约 Agent、RAG 项目业务独立：不接入真实订票、酒店、价格或可用性接口，也不会执行外部写入。

## 功能闭环

```text
用户旅行请求
  -> 结构化需求（出发地、目的地、日期、天数、人数、预算、偏好）
  -> 信息不足：clarify
  -> 信息齐全：search_trip_options
  -> 版本化 demo_catalog_v1
  -> 含来源说明的 final_plan
```

- 强类型 JSON 合同：`clarify`、`tool_call`、`final_plan`。
- 受限工具边界：解析层不编造价格、交通、酒店可订状态；旅行目录是版本化本地样例。
- 数据管线：配置驱动生成 480/80/120 条训练、验证、冻结评测 split，记录 SHA-256、覆盖率、跨 split 文本隔离与待人工复核队列。
- 三种显式推理 profile：`rule` 用于离线合同回归，`hf_adapter` 加载 Transformers/PEFT adapter，`llama_cpp` 加载 Q4_K_M 基座与 LoRA GGUF；三者共用同一个 `TravelExtractor`/orchestrator 合同。

## 已真实执行的实验

### Qwen3-1.7B 4-bit QLoRA SFT

- 在 RTX 4060 Laptop GPU 上完成 4-bit NF4、LoRA-r16 的 SFT：480 条合成训练样本、80 条验证样本、4 个 epoch；`run_manifest.json` 保存模型、数据、配置、依赖、GPU、训练参数和 adapter 哈希。
- 在同一份 120 条版本化冻结集、相同 prompt 与确定性解码条件下，基座模型的合同合法率与动作匹配均为 0；SFT adapter 的 JSON 合法率、合同合法率和动作匹配均为 100%，槽位 micro-F1 为 99.88%，工具参数精确匹配为 98.53%。
- 该质量评测的 adapter P50/P95 生成延迟为 22.89/31.91 秒，平均输出 144.3 token。它反映的是 Transformers 4-bit 全量生成条件，不应与下面的 llama.cpp 单次部署基准混用。

### llama.cpp 本地部署

- 将公开 Qwen3-1.7B 基座转换后的 Q4_K_M GGUF（1.28 GB）与本项目 SFT LoRA F16 GGUF（34.9 MB）通过 `llama-cli --lora` 组合加载；部署清单记录两份 GGUF、prompt 与冻结集的 SHA-256。
- 在 llama.cpp `b10545`、`-ngl 99`、`temperature=0`、`reasoning=off` 条件下，对 3 条选定冻结样例完成实际 one-shot CLI smoke。原始 JSON 合法率为 100%，原始合同合法率为 66.7%；其中 1 条仅遗漏 `tool_call.arguments`，部署适配层只从模型已输出且完整的 `request` 机械复制工具参数，修复后合同、动作和工具参数匹配均为 100%。
- 这组 3 样例 one-shot 测量（包含进程启动和模型加载）的 P50/P95 为 4.42/4.66 秒，模型解码 P50 为 92.2 token/s。它是部署 smoke，不是全量质量结论。

### GGUF 应用接入、全量质量与冷暖性能

- 实际 application smoke 选取一条缺字段和一条完整冻结样本，均通过 `llama_cpp extractor → 严格合同 → orchestrator`；分别得到 `clarify` 与 `tool_call → demo_catalog_v1 → final_plan`，2/2 通过。模型输出无权生成预订、支付、实时价格或外部写入。
- 在同一 120 条冻结合成集上对 GGUF + LoRA profile 独立评测：raw 合同/动作/槽位 F1/工具参数 EM 为 68.33%/60.83%/79.21%/50.00%；18 条仅缺 `tool_call.arguments` 的输出经允许的机械复制后为 83.33%/75.83%/90.16%/76.47%。该结果低于 HF adapter，不能用来替代上方 99.88%/98.53% 指标。
- 冷启动以 20 个独立 `llama-cli` 进程计时，20/20 成功，P50/P95 为 3.54/4.25 秒；常驻 `llama-server` 完成 3 次 warm-up 后顺序测量 100 次，100/100 请求成功，P50/P95 为 0.98/1.71 秒，生成速度 P50 109.5 token/s。两者都不是并发压测或生产 SLA。

所有结果的原始输出和逐 case 记录分别位于：

- `reports/training/qwen3-1.7b-travel-sft-v1.run_manifest.json`
- `reports/model_eval/qwen3-1.7b-base__travel-eval-v1.0.json`
- `reports/model_eval/qwen3-1.7b-travel-sft-v1__travel-eval-v1.0.json`
- `reports/deployment/llama_cpp_q4km_lora_v1.json`
- `reports/application/llama_cpp_application_smoke_v1.json`
- `reports/deployment/llama_cpp_full_frozen_eval_v1.json`
- `reports/deployment/llama_cpp_performance_v1.json`
- `reports/release/travel_resume_release_gate_v1.json`

## 复现

基础离线测试：

```powershell
uv sync --frozen
uv run --no-sync pytest -q
uv run --no-sync travel-itinerary demo --today 2026-08-20 --extractor-mode rule --message "从南京去杭州，10月1日出发玩三天，两个人，预算5000元，喜欢人文和美食。"
uv run --no-sync travel-itinerary audit-data
```

重新运行 SFT 前，显式指定本地基础模型路径：

```powershell
$env:TRAVEL_BASE_MODEL_PATH = (Resolve-Path "models/base/qwen3-1.7b-hf").Path
uv run --no-sync python -m travel_itinerary.training --config configs/training/qwen3_1_7b_travel_sft_v1.json --validate-only
uv run --no-sync python -m travel_itinerary.training --config configs/training/qwen3_1_7b_travel_sft_v1.json
```

重新运行本地 GGUF smoke benchmark：

```powershell
$env:TRAVEL_LLAMA_CLI_PATH = "<llama-cli.exe的本地路径>"
uv run --no-sync python -m travel_itinerary benchmark-llama-cpp
```

应用接入、全量 GGUF 质量、冷暖性能与最终证据门禁：

```powershell
uv run --no-sync travel-itinerary smoke-llama-application --llama-cli $env:TRAVEL_LLAMA_CLI_PATH
uv run --no-sync travel-itinerary evaluate-llama-cpp --llama-cli $env:TRAVEL_LLAMA_CLI_PATH
uv run --no-sync travel-itinerary benchmark-llama-runtime `
  --llama-cli $env:TRAVEL_LLAMA_CLI_PATH `
  --llama-server $env:TRAVEL_LLAMA_SERVER_PATH
uv run --no-sync python -m travel_itinerary.evidence verify --require-ready
```

## 证据边界

- 数据是配置驱动生成并经过程序化审计的合成数据，不是生产用户数据；30 条人工复核样本仍标记为 `pending_human_review`。
- DPO 偏好对只做了数据准备，未执行 DPO 训练，因此项目不声称已完成 DPO。
- 冻结集指标证明该版本化合成合同上的结果，不代表真实旅行推荐质量、真实用户效果或线上业务指标。
- `demo_catalog_v1` 仅用于本地工具链验证；项目不下单、不处理支付或敏感身份信息。
- HF adapter 与 GGUF 是不同推理 profile：简历中的 100%/99.88%/98.53% 来自 HF adapter；GGUF 的 120-case 结果按上文单独披露。

详细实验约束见 [experiment-protocol.md](docs/experiment-protocol.md)，本地部署口径见 [local-deployment.md](docs/local-deployment.md)。
