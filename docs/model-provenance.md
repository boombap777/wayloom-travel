# 基础模型标注更正（2026-09-08）

实际用于此次 QLoRA 的基础检查点是 **Qwen/Qwen3-1.7B**，不是单独的
`Qwen/Qwen3-1.7B-Base` 仓库。原训练 JSON 报告中的 model_id 是配置填写错误，
不能仅凭那个字符串确定权重来源。

证据：本地 Hugging Face 下载元数据记录 revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`。官方固定版本 API 的两个权重分片
名称、大小和 SHA-256 均与原训练报告完全一致。可独立核对：

- [官方固定版本元数据](https://huggingface.co/api/models/Qwen/Qwen3-1.7B/revision/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e?blobs=true)
- `configs/models/qwen3_1_7b_origin.json` 保存该最小核对记录。
- `python -m travel_itinerary.model_origin --model-dir <模型目录>` 校验实际本地字节。

当前训练配置与发布模型卡已改用正确 ID；历史训练报告、评测结果与权重不改写。
“基座与适配器对比”指本次任务 QLoRA 前后的对比，不是声称从纯预训练 Base
检查点开始。被冻结的 120-case 分数仍属于那次实验，不代表任意自然语言泛化能力。
