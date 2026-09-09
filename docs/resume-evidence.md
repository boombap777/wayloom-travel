# 简历证据包：旅行行程需求解析与规划助手

本文件只汇总当前仓库已经实际运行、可回溯的事实，供校招项目经历撰写和面试复盘使用。

## 可直接写入简历的版本

**旅行行程需求解析与规划助手（本地后训练 PoC）** ｜ Agent / 本地模型应用开发

面向中文旅行需求的结构化解析与行程草案生成，构建“需求决策—受限工具—本地目录—草案返回”闭环；以版本化合成数据验证本地 Qwen3 后训练和 GGUF 部署能力。

- **结构化决策与工具边界：** 定义 `clarify`、`tool_call`、`final_plan` 强类型 JSON 合同；仅在出发地、目的地、日期、天数和人数齐全时调用 `search_trip_options`，目录工具固定为本地 `demo_catalog_v1`，避免模型把演示数据表述为实时票价或可订库存。
- **数据与评测闭环：** 配置化生成 480/80/120 条训练、验证、冻结评测 split，记录 SHA-256、场景覆盖和跨 split 文本隔离；保留 30 条 `pending_human_review` 样本，并以 51 个离线单元/集成测试覆盖合同、工具、数据审计、模型输出、应用路由和部署证据逻辑。
- **本地 QLoRA 后训练：** 在 RTX 4060 Laptop GPU 上对 Qwen3-1.7B 执行 4-bit NF4、LoRA-r16 SFT（480 条训练、80 条验证、4 epoch）；在 120 条版本化合成冻结集上，合同合法率与动作匹配由基座的 0 提升至 100%，槽位 micro-F1 为 99.88%，工具参数精确匹配为 98.53%。
- **GGUF 部署与合同修复：** 将 Q4_K_M 基座（1.28 GB）与 34.9 MB LoRA GGUF 通过 llama.cpp `--lora` 加载；在 3 条冻结 smoke 中，原始输出均为可解析 JSON，针对 1 条仅缺少工具参数的结果从其完整 `request` 机械构造 arguments，修复后 3/3 合同与动作匹配、2/2 工具参数匹配；one-shot P50/P95 为 4.42/4.66 s，解码 P50 为 92.2 token/s。

技术栈：Python / Qwen3-1.7B / Transformers / PEFT / bitsandbytes / QLoRA / GGUF / llama.cpp / pytest

## 面试时应主动说明的边界

- 480/80/120 是程序化审计的合成旅行样本规模，不是生产用户数据；人工复核还未完成。
- 120 条冻结集指标说明的是固定结构化合同上的离线结果，不是真实旅行推荐、订单转化或线上准确率。
- llama.cpp 的 4.42/4.66 s 是 3 条样例的 one-shot CLI 时间，包含进程启动和模型加载；不能写成常驻服务 P95，更不能写成整体模型延迟。
- 全量 GGUF 的修复后合同/动作/槽位 F1/工具参数 EM 为 83.33%/75.83%/90.16%/76.47%，明显低于 HF adapter；这组数据用于说明量化部署边界，不能与简历中的 HF 指标合并。
- 新的 cold 20 次 P95 为 4.25 s，warm 100 次 P95 为 1.71 s；简历保留的 4.66 s/92.2 token/s 是原始 3-case Smoke 的精确口径，不应悄悄替换成更好的新数字。
- DPO 数据已准备但未实际训练，不能写“完成 DPO”。
- 本地目录不是真实供应商、票务或酒店接口；项目没有下单、支付或外部写入。

## 证据索引

| 主张 | 权威产物 |
| --- | --- |
| SFT 配置、GPU、版本、参数、耗时与 adapter 哈希 | `reports/training/qwen3-1.7b-travel-sft-v1.run_manifest.json` |
| 基座与 adapter 的 120 条冻结集原始输出和指标 | `reports/model_eval/*.json` |
| GGUF 哈希、llama.cpp 版本、3 条部署 smoke 原始 stdout 和指标 | `reports/deployment/llama_cpp_q4km_lora_v1.json` |
| 两条实际 model-to-tool-to-plan 应用闭环 | `reports/application/llama_cpp_application_smoke_v1.json` |
| GGUF 全部 120 条 raw/repaired 质量和逐 case 错误 | `reports/deployment/llama_cpp_full_frozen_eval_v1.json` |
| 20 次 cold 与 100 次 post-warm-up 常驻性能 | `reports/deployment/llama_cpp_performance_v1.json` |
| 当前源码/模型/数据/报告哈希与最终 claim 状态 | `reports/release/travel_resume_release_gate_v1.json` |
| 数据规模、split 隔离、SHA-256 和人工复核状态 | `data/manifests/travel-v1.0.manifest.json`、`data/review/v1.0/pending_human_review.jsonl` |
