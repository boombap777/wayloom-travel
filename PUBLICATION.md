# 发布与复现

本项目是个人本地技术原型，不是订票网站。自有代码 MIT；Qwen 模型与派生模型包
保留 Apache-2.0 许可和来源说明。源码包、模型包分开发布，禁止将大模型直接加入普通 Git。

## 一分钟浏览器演示

安装 Python 3.12 和 uv，从仓库根目录执行：

```shell
uv sync --frozen
uv run --no-sync travel-itinerary-web
```

访问 http://127.0.0.1:8504/。点击“还没想好”，提交后补充
`从南京出发，10月1日，玩三天，两个人，预算5000元。`，即可看到目录草案。
“调整需求”会启动新的完整请求，不会把旧条件隐式混入新计划。
`--port 8514` 可更换端口。只监听本机，无登录系统，**不应直接暴露到公网**。
页面与 API 不接收任意模型路径、执行命令或工具指令；模型由启动参数决定。
上下文仅在本页内存中保存，刷新或重新开始即清空；最多四次补充，总计6000字。

## 默认测试

```shell
uv run --no-sync pytest --cov=travel_itinerary --cov-report=term
```

源码内包含版本化合成训练/验证/冻结数据，以及 v1.0 预处理数据，不要求现成的本地模型。
测试覆盖真实 HTTP 请求、补全到草案、会话隔离、来源标记、非法字段、跨源/Host、
超长输入、模型无效输出和忙碌状态。模型失败返回明确错误，绝不偷偷回退到规则模式。

## 真实模型模式

### llama.cpp

将配套模型包解压到仓库根目录，保留 `models/gguf/` 和 `models/adapters/` 层级。
模型包不是源码的一部分；请从 [v0.1.0 Release](https://github.com/boombap777/wayloom-travel/releases/tag/v0.1.0)
下载已发布的配套资产，并核对该页面与包内清单提供的 SHA-256。
或自行使用以下 HF 原始检查点与本项目的 LoRA 训练/转换脚本生成对应 GGUF。
自行准备官方 llama.cpp `llama-cli`；历史实验版本为 build 10545 / `a30273376`。

```powershell
$env:TRAVEL_LLAMA_CLI_PATH = '<llama-cli.exe路径>'
uv run --no-sync travel-itinerary-web --extractor-mode llama_cpp
```

网页会显示真实选择的模型模式。首次推理失败会明确报错，不展示虚构方案。
运行文件与模型哈希以模型包内 `MODEL_ASSETS.json` 为准；新硬件不承诺相同延迟。

### HF adapter / 复训

需要兼容 CUDA 的 GPU 环境。历史锁定依赖见 `requirements-train.txt`，不属于无模型安装。
按硬件准备对应 PyTorch CUDA 构建，例如历史 cu128 环境：

```shell
uv pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-train.txt
uv run --no-sync hf download Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --local-dir models/base/qwen3-1.7b-hf
uv run --no-sync python -m travel_itinerary.model_origin --model-dir models/base/qwen3-1.7b-hf
```

下载约4 GB 权重，需网络；完整性校验不得跳过。Windows 示例：

```powershell
$env:TRAVEL_BASE_MODEL_PATH = (Resolve-Path 'models/base/qwen3-1.7b-hf').Path
uv run --no-sync python -m travel_itinerary.training --config configs/training/qwen3_1_7b_travel_sft_v1.json --validate-only
uv run --no-sync travel-itinerary-web --extractor-mode hf_adapter
```

若不使用配套 adapter，可在**新的实验目录/源码副本**中执行训练命令，避免覆盖已有 adapter：
`uv run --no-sync python -m travel_itinerary.training --config configs/training/qwen3_1_7b_travel_sft_v1.json`。
GGUF 转换在 llama.cpp 的固定版本源码中使用 `convert_hf_to_gguf.py`、`llama-quantize`
和 `convert_lora_to_gguf.py` 完成；需保持同一基础检查点，不能用另一个 Base 变体代替。
源码包提供的直接演示不要求访客重新训练或转换模型。

## 指标边界与历史报告

- HF adapter 的120条冻结合成集：合同/动作100%、槽位F1 99.88%、工具参数EM 98.53%。
- GGUF 三条 smoke：原始合同2/3，一条仅补齐缺失工具参数后3/3；P95 4.66秒含加载。
- GGUF 全量120条与 HF 明显不同：修复后合同83.33%、动作75.83%，不能沿用HF高分。
- 目录只有五个城市样例，费用是固定演示值；活动大纲不是按任意天数生成的完整路线。
- `reports/training`、`reports/model_eval`、`reports/deployment` 保存历史合成实验原始记录。
  其中旧目录是实验来源信息，不是当前机器的启动配置；源码发布不包含原始运行日志。
  历史 `model_id` 的更正见 `docs/model-provenance.md`，不改写原始分数或测量时间。
- `evidence verify --require-ready` 是完整本地实验制品门禁，需额外模型和当前证据清单。
  普通无模型 CI 不应伪称重新完成训练或全量模型评测。

## 打包

`python scripts/publication.py --output <新目录>` 导出源码与清单，不复制 Git 历史、不提交。
`python scripts/export_model_assets.py --output <另一新目录> --include-base` 独立准备模型包。
请只把源码目录用于 Git；模型压缩包用于 Release 或模型仓库，保留许可和哈希清单。
