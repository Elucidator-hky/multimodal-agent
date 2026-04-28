# 多模态客服智能体(DataFountain 大奖赛)

参加 DataFountain "具有多模态能力的客服智能体设计" 比赛。

- 比赛:https://www.datafountain.cn/competitions/1165
- 任务:基于 22 份产品手册(含 2608 张插图)+ 通用客服话术,构建多模态 RAG 客服 agent
- 输出:400 题 csv 提交(A 榜)+ 决赛 REST API(`/chat`)
- 框架:**LangGraph** 编排 + **LlamaIndex** 检索 + **bge-m3** embedding + **qwen-plus** LLM

详细进度见 [PROGRESS.md](PROGRESS.md)。

## 架构

```
START → router ─┬→ rag_search → generate ─┐
                └→ service_gen ────────────┴→ format_output → END
                  ↑               ↑                 ↑
              客服话术         产品答案+图        官方格式输出
```

5 节点全部完成,端到端 eval 通过(34 题 / 路由 100% / 平均 4.6s)。

## 在新机器上跑起来

### 1. 装依赖

```bash
git clone <repo-url> multimodal-agent
cd multimodal-agent

# 基础依赖
pip install -r requirements.txt

# PyTorch(关键!根据你的硬件选)
# GPU(NVIDIA,推荐):
pip install torch --index-url https://download.pytorch.org/whl/cu126
# 或 CPU only(慢 10x):
# pip install torch
```

### 2. 配置 API key

```bash
cp .env.example .env
# 编辑 .env,填入你的 QWEN_API_KEY(阿里云 DashScope 申请)
```

### 3. 数据情况

仓库**已包含**:
- 赛题数据 `data/KnowledgeBase/`(22 手册 + 2608 张插图)
- LLM 切块缓存 `data/chunks_llm/`(22 个 JSON,2330 sections)
- 题目 `data/question-public.csv`

仓库**未包含**(自动生成 / 下载):
- `models/` —— bge-m3(~4.3GB),首次跑 `python -m src.knowledge_base` 时自动从 HuggingFace 下载(用 HF_ENDPOINT=hf-mirror.com 镜像加速,中国大陆可用)
- `index/` —— FAISS 索引,本地构建

### 4. 建索引(首次必跑,~5-10 分钟 GPU / ~30 分钟 CPU)

```bash
HF_ENDPOINT=https://hf-mirror.com python -m src.knowledge_base
```

输出:
- 自动下载 bge-m3 到 `models/`
- 用 LLM 切好的 chunks 跑 embedding
- 写到 `index/`(2330 个向量,1024 维,~13MB)

### 5. 验证 agent 跑通

```bash
# 端到端测试 4 道题(2 产品 + 2 客服)
python -m src.agent_graph

# 跑 30 题 RAG eval
python evals/run_rag_eval.py

# 跑 34 题端到端 eval
python evals/run_e2e_eval.py
# 报告:evals/output/e2e_eval.md
```

## 项目结构

```
multimodal-agent/
├── config.py                       # 集中配置(从 .env 读 key)
├── .env.example                    # 配置模板
├── requirements.txt
├── PROGRESS.md
│
├── src/
│   ├── state.py                    # LangGraph State 定义
│   ├── agent_graph.py              # 5 节点整图组装(主入口)
│   ├── knowledge_base.py           # 索引构建 + 检索接口
│   ├── llm_chunker.py              # LLM 切块器(预处理用,不在运行时调)
│   ├── reranker.py                 # bge-reranker(默认不启用)
│   ├── tracing.py                  # 节点级 trace 工具(零侵入)
│   └── nodes/
│       ├── router.py               # product / service 二分类
│       ├── service_gen.py          # 客服话术
│       ├── rag_search.py           # 检索 top-5 chunks
│       ├── generate.py             # 产品答案 + 图片 ID
│       └── format_output.py        # 拼成官方提交格式
│
├── scripts/                        # 一次性预处理脚本(已跑过,新机器无需再跑)
│   ├── cache_all_chunks.py         # 批量 LLM 切块
│   ├── process_english_manual.py   # 英文手册粗切+细切
│   ├── patch_uncovered_gaps.py     # 补漏 gap(LLM 偶发漏抄)
│   ├── patch_english_part3_simple.py  # 段 3 按 # 切(LLM 失败兜底)
│   ├── patch_missing_pics.py       # 补 PIC(已弃用)
│   ├── rerun_quality_failed.py     # 重跑质量不达标的手册
│   ├── rerun_english_part.py       # 重跑英文某段
│   └── auto_label_keywords.py      # LLM 标注 RAG eval keywords
│
├── evals/
│   ├── rag_cases.py                # 30 题 RAG eval 用例 + evaluator
│   ├── rag_keywords.json           # LLM 自动标注的关键词
│   ├── run_rag_eval.py             # RAG 检索 eval
│   ├── run_e2e_eval.py             # 端到端 eval
│   ├── run_router.py               # router 节点 eval
│   ├── run_service_gen.py          # service_gen 节点 eval
│   └── output/                     # 报告(本地生成,不 commit)
│
├── data/
│   ├── question-public.csv
│   ├── submission-example.csv
│   ├── KnowledgeBase/手册/         # 22 份手册 + 插图(已 commit)
│   └── chunks_llm/                 # LLM 切块缓存(已 commit,~1.8MB)
│
├── history_code/                   # baseline 老代码(参考)
└── models/   index/   evals/output/   .env   # 不 commit
```

## 评分标准

- 初赛 = 系统设计 30% + 技术实现 70%
- 技术实现 = LLM 裁判按 1-5 分自动打分(看图文配合 + 回答质量 + 结构)
- 提交格式:
  - 产品题:`"文本(含<PIC>)", ["图片id1", "图片id2"]`
  - 客服题:`"纯文本"`

## 已知限制 / 下一步

- **空调 vs 空气净化器歧义**:同主题章节跨产品手册混淆(块级 Recall@1 = 76.7%)。Round 2 解法:router 输出具体产品名 + rag_search metadata 过滤
- **未实现**:跑全量 400 题生成 A 榜 csv(脚本待写)
- **未实现**:决赛 REST API(`/chat` 端点 + streaming)
