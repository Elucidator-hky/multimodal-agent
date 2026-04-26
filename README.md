# 多模态客服智能体（DataFountain 大奖赛）

参加 DataFountain "具有多模态能力的客服智能体设计" 比赛的项目代码。

- 比赛链接：https://www.datafountain.cn/competitions/1165
- 任务：基于 22 份产品手册（含 2608 张插图）+ 通用客服话术，构建多模态 RAG 客服 agent
- 输出：400 题 csv 提交（A 榜）+ 决赛 REST API（`/chat`）
- 框架：**LangGraph**（节点+边的图编排）+ **Qwen-Plus**（阿里云 DashScope）

## 当前进度

详见 [PROGRESS.md](PROGRESS.md)。

简要：
- ✅ 数据/题目分析完成
- ✅ LangGraph hello world 跑通
- ✅ State 设计完成
- ✅ `service_gen` 节点完成（客服话术生成，含 prompt + 12 题 eval）
- ⬜ `router` / `rag_search` / `generate` / `format_output` 节点待写
- ⬜ 完整 graph 组装待做
- ⬜ 跑全量 400 题 + 提交 A 榜

## 在新机器上跑起来

```bash
# 1. clone 代码
git clone <repo-url> multimodal-agent
cd multimodal-agent

# 2. 安装依赖
pip install -r requirements.txt
pip install langgraph

# 3. 配置 API key
cp .env.example .env
# 编辑 .env，填入你的 QWEN_API_KEY（DashScope 申请）

# 4. 下载赛题数据
# 从 https://www.datafountain.cn/competitions/1165/datasets 下载：
#   - KownledgeBase.zip → 解压到 data/KnowledgeBase/
#   - question-public.csv → 放到 data/
#   - submission-example.csv → 放到 data/

# 5. 构建 FAISS 索引（首次运行）
python -m src.knowledge_base
# 索引会保存到 index/

# 6. 跑 hello world 验证 LangGraph 装好了
python src/hello_graph.py

# 7. 跑 service_gen 节点 eval
python evals/run_service_gen.py
```

## 项目结构

```
multimodal-agent/
├── config.py                    # 配置（从 .env 读敏感信息）
├── .env                         # 真实 API key（不提交）
├── .env.example                 # 配置模板
├── requirements.txt
├── PROGRESS.md                  # 当前进度记录
│
├── src/
│   ├── state.py                 # LangGraph State 定义（共享）
│   ├── knowledge_base.py        # FAISS 索引 + 手册解析（baseline 复用）
│   ├── hello_graph.py           # LangGraph 入门 demo
│   └── nodes/
│       ├── __init__.py
│       └── service_gen.py       # 客服话术节点（已完成）
│
├── evals/
│   ├── run_service_gen.py       # 12 题 eval（3 官方 + 9 csv）
│   └── output/                  # 评估结果（不提交）
│
├── history_code/                # baseline 旧代码（参考用，已不直接使用）
│   ├── api.py                   # 旧 FastAPI
│   ├── rag.py                   # 旧 RAG 引擎
│   ├── generate_submission.py   # 旧批量提交脚本
│   ├── agent_graph.py           # 第一版 LangGraph 整图（未跑通）
│   └── test_service_gen.py
│
└── data/                        # 赛题数据（不提交，新机器重新下载）
    ├── question-public.csv
    ├── submission-example.csv
    └── KnowledgeBase/手册/
        ├── *.txt                # 22 份产品手册
        └── 插图/                 # 2608 张图片
```

## Agent 架构（设计中）

```
       START
         ↓
     [router]                ← 意图分析：产品题 or 客服题
       ↙   ↘
[rag_search]  [service_gen]   ← 条件边分流
       ↓             ↓        ← service_gen 已完成 ✅
   [generate]        ↓
       ↓             ↓
   [format_output] ← ┘        ← 拼成官方格式 "文本", [图片ID]
         ↓
        END
```

## 评分标准提醒

- 初赛 = 系统设计 30% + 技术实现 70%
- 技术实现 = LLM 裁判按 1-5 分自动打分（看图文配合 + 回答质量 + 结构）
- 提交格式：
  - 产品题：`"文本（含<PIC>）", ["图片id1", "图片id2"]`
  - 客服题：`"纯文本"`
