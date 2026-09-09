# 数据血缘与质量边界

`travel-v1.0` 是由 `configs/data/travel_v1.0.json` 控制的、确定性合成数据集，不包含真实用户对话、订单、支付或平台爬取数据。

生成命令会创建：

- `data/raw/v1.0/train.jsonl`：训练原始样本；
- `data/raw/v1.0/validation.jsonl`：验证原始样本；
- `data/eval/v1.0/frozen_cases.jsonl`：冻结评测样本；
- `data/processed/v1.0/*`：SFT 与 DPO 渲染结果；
- `data/manifests/travel-v1.0.manifest.json`：规模、覆盖、SHA-256、隔离检查；
- `data/review/v1.0/pending_human_review.jsonl`：待人工抽检队列。

生成器使用不同 split 的模板族，并以归一化后的用户文本执行跨 split 重复检测。它还会验证：每条期望输出能构造为强类型决策、没有预约域残留词、并与规则基线合同一致。

这是一套**合成数据的程序质检**，不等于真实用户分布验证，也不等于人工标注或人工审核。简历或面试中只能写“构造并程序审计合成旅行对话数据集”；除非有人在 review 队列中留下审核人、日期和结论，否则不能写“人工审核”。

