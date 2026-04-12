# OxyGent-SFT: Self-Evolving Multi-Agent System with SFT Flywheel & RAG

<p align="center">
  <a href="https://github.com/jd-opensource/OxyGent">
    <img src="https://img.shields.io/badge/Upstream-jd--opensource/OxyGent-blue.svg" alt="upstream"/>
  </a>
  <a href="https://github.com/jd-opensource/OxyGent/blob/v4/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="license"/>
  </a>
</p>

> **本项目基于京东开源多智能体框架 [OxyGent](https://github.com/jd-opensource/OxyGent) 进行扩展。** OxyGent 将工具、模型、智能体统一为可插拔的原子算子（Oxy），支持模块化组装、动态规划与弹性扩展。本项目在此基础上构建了完整的 SFT 飞轮闭环与 RAG 知识增强系统，使智能体具备自我迭代进化能力。

---

## 项目动机

原版 OxyGent 提供了强大的多智能体协作框架，但存在两个关键缺口：

1. **没有反馈闭环**：智能体在线运行后，无法从用户反馈中学习，模型无法自我进化
2. **RAG 仅有框架定义**：`RAGAgent` 类存在，但缺少完整的知识库构建、向量化存储与检索实现

本项目围绕这两个缺口，构建了 **SFT 飞轮** 和 **RAG 知识增强** 两大系统，形成从推理到进化的完整闭环。

---

## 核心贡献

### 一、SFT 飞轮闭环系统

构建了自我迭代的 SFT 飞轮，使智能体能够从用户反馈中持续学习：

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ 智能体   │────▶│ 人工打分 │────▶│ 数据提取 │────▶│ LoRA微调 │
│ 在线运行 │     │ 评分系统 │     │ 自动评分 │     │ 模型训练 │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
      ▲                                                 │
      └─────────────────────────────────────────────────┘
                    微调后模型重新部署
```

#### 1.1 评分系统（Rating System）

**后端 API**（`oxygent/web/rating_api.py`）：

| 接口 | 方法 | 功能 |
|------|------|------|
| `/api/rate` | POST | 提交评分（1-5分 + 多维度评分 + 评论） |
| `/api/rate/{trace_id}` | GET | 查询指定 trace 的评分记录 |
| `/api/ratings/stats` | GET | 聚合统计（总数、均分、分布） |

**前端组件**（`oxygent/web/js/rating_widget.js`）：

- 五星评分，悬停动画效果
- 快捷标签按钮组：回答准确 / 步骤清晰 / 信息过时 / 遗漏关键点 / 工具调用错误 / 超出预期
- 可折叠评论输入框，评分 ≤ 2 分时自动展开
- 加载中 / 已提交确认状态
- 仅在 `type=="answer"` 消息后显示
- 挂载时自动查询历史评分并回显

#### 1.2 训练数据提取流水线（`tools/extract_training_data.py`）

**三级混合打分体系**：

| 打分方式 | 成本 | 触发条件 | 权重 |
|---------|------|---------|------|
| 人工评分 | 高 | 用户主动评分 | 100%（优先） |
| 规则打分 | 零 | 始终执行 | 40% |
| LLM 裁判打分 | 中 | 规则分 ≥ 1.5 且配置了 judge-api | 60% |

**规则打分扣分项**：
- 无最终回答 → 0.5 分
- 回答长度 < 15 字 → 扣 2.0
- 包含失败关键词 → 扣 1.5
- 工具执行失败率 → 最多扣 2.0
- 工具调用次数 > 15 → 扣 1.5

**输出格式**：ChatML 训练格式，保留完整工具调用链，自动划分 train/val/test 集。

#### 1.3 LoRA 微调训练（`tools/lora_finetune.py`）

三个子命令：

```bash
# 训练（默认 QLoRA 4bit）
python tools/lora_finetune.py train \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --data-dir ./sft_data \
  --output-dir ./lora_output

# 合并适配器到基础模型
python tools/lora_finetune.py merge \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --adapter-path ./lora_output \
  --output-path ./merged_model

# 部署微调后的模型
python tools/lora_finetune.py deploy \
  --model-path ./merged_model
```

**训练配置**：
- 默认 QLoRA 4bit（BitsAndBytesConfig + NF4 量化）
- LoRA rank=16, alpha=32, dropout=0.05
- 目标模块：q_proj, k_proj, v_proj, o_proj
- 支持 bf16 全精度模式（`--no-4bit`）
- 支持从已有 adapter 继续训练（`--resume-adapter`）

---

### 二、RAG 知识增强系统

补全了 OxyGent `RAGAgent` 从数据采集到检索增强的完整链路：

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ 网页爬虫 │────▶│ 智能分块 │────▶│ LLM API  │────▶│ OxyGent  │
│ (crawl)  │     │ (chunk)  │     │ Embedding│     │ RAGAgent │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
```

#### 2.1 知识采集（`tools/crawl_jd_culture.py`）

- 异步爬取京东官网企业文化页面
- 智能文本分块：句号/问号/感叹号断句 + 滑动窗口重叠（500 字/块，50 字重叠）
- 输出结构化知识块 JSON

#### 2.2 向量化存储（`tools/vectorize_knowledge.py`）

- **默认使用 OpenAI Embedding API**（`text-embedding-3-small`，1536 维）
- 支持切换到本地模型（`--local-model BAAI/bge-small-zh-v1.5`）
- 本地向量存储：`index.json`（元数据）+ `vectors.npy`（向量矩阵）
- 余弦相似度检索，支持 Top-K 排序

```bash
python tools/vectorize_knowledge.py \
  --chunks-file knowledge_data/knowledge_chunks.json \
  --output-dir vector_store \
  --api-key $OPENAI_API_KEY \
  --test
```

#### 2.3 RAG 集成（`tools/rag_demo.py`）

```python
from oxygent import MAS, RAGAgent, OpenAILLM
from tools.rag_demo import retrieve_jd_knowledge

llm = OpenAILLM(name="default_llm", model_name="gpt-4o-mini")

rag_agent = RAGAgent(
    name="jd_assistant",
    llm_model="default_llm",
    prompt="根据知识库回答：${knowledge}",
    func_retrieve_knowledge=retrieve_jd_knowledge
)

async with MAS(name="rag_demo", oxy_space=[llm, rag_agent]) as mas:
    response = await mas.chat_with_agent(payload={"query": "京东的企业文化是什么？"})
```

#### 2.4 一键部署（`tools/setup_jd_rag.py`）

```bash
python tools/setup_jd_rag.py --api-key $OPENAI_API_KEY
```

---

## 项目结构（新增部分）

```
OxyGent/
├── oxygent/
│   ├── web/
│   │   ├── rating_api.py           # 评分系统后端 API
│   │   └── js/
│   │       └── rating_widget.js    # 前端评分组件
│   └── routes.py                   # 挂载评分路由
├── tools/
│   ├── crawl_jd_culture.py         # 京东企业文化爬虫
│   ├── vectorize_knowledge.py      # 知识库向量化（OpenAI API / 本地模型）
│   ├── rag_demo.py                 # RAG 集成示例
│   ├── setup_jd_rag.py             # RAG 一键部署
│   ├── extract_training_data.py    # 训练数据提取流水线
│   └── lora_finetune.py            # LoRA 微调（train/merge/deploy）
└── requirements.txt                # 更新后的依赖
```

---

## 与原版 OxyGent 对比

| 维度 | 原版 OxyGent | 本项目 |
|------|-------------|--------|
| 多智能体推理 | ✅ | ✅ 不变 |
| 用户反馈 | ❌ 无评分系统 | ✅ 五星评分 + 标签 + 评论 |
| 训练数据提取 | ❌ 无 | ✅ ES → ChatML + 混合打分 |
| 模型微调 | ❌ 无 | ✅ QLoRA 训练 + 合并 + 部署 |
| RAG 实现 | ⚠️ 仅有 RAGAgent 框架 | ✅ 完整链路：爬取→分块→向量化→检索 |
| Embedding | ❌ 无实现 | ✅ OpenAI API / 本地模型双模式 |
| 自我进化 | ❌ 无闭环 | ✅ 推理→评分→数据→训练→部署→循环 |

---

## 快速开始

### 环境准备

```bash
pip install -r requirements.txt
```

### RAG 知识增强

```bash
# 设置 API Key
export OPENAI_API_KEY=sk-xxx

# 一键部署
python tools/setup_jd_rag.py --api-key $OPENAI_API_KEY

# 或分步执行
python tools/crawl_jd_culture.py
python tools/vectorize_knowledge.py --chunks-file knowledge_data/knowledge_chunks.json --output-dir vector_store --api-key $OPENAI_API_KEY --test
python tools/rag_demo.py --mode simple
```

### SFT 飞轮

```bash
# 1. 启动 OxyGent 服务（评分 API 自动挂载）
python demo.py

# 2. 在前端对智能体回答进行评分

# 3. 提取训练数据
python tools/extract_training_data.py \
  --app my_app \
  --es http://localhost:9200 \
  --out ./sft_data \
  --min-score 3.5

# 4. LoRA 微调
python tools/lora_finetune.py train \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --data-dir ./sft_data \
  --output-dir ./lora_output

# 5. 合并 & 部署
python tools/lora_finetune.py merge --base-model ... --adapter-path ./lora_output --output-path ./merged
python tools/lora_finetune.py deploy --model-path ./merged
```

---

## 致谢

- [OxyGent](https://github.com/jd-opensource/OxyGent) — 京东开源的多智能体协作框架，本项目的基础
- [PEFT](https://github.com/huggingface/peft) — HuggingFace 参数高效微调库
- [TRL](https://github.com/huggingface/trl) — HuggingFace 强化学习训练库

## 许可证

本项目遵循 [Apache License 2.0](./LICENSE)，与原版 OxyGent 保持一致。
