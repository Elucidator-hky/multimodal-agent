# 项目进度

更新时间:2026-04-28

## 已完成

### 1. 数据 / 题目分析
- 22 份产品手册(20 中文 + 1 汇总英文 + 1 英文摄像)+ 2608 张插图
- 400 题分布:通用客服 ~37(9%) / 中文产品 ~162(41%) / 英文产品 ~187(47%) / 边界 ~14(3%)
- 提交格式:产品题 `["文本(<PIC>)", ["id1","id2"]]` / 客服题 `"纯文本"`

### 2. 技术栈
| 层 | 用什么 |
|---|---|
| 流程编排 | **LangGraph** 1.1(节点 + 条件边) |
| RAG 框架 | **LlamaIndex** 0.14(只用检索部分,生成自己写) |
| Embedding | **BAAI/bge-m3**(多语言,1024 维,GPU fp16) |
| 向量库 | FAISS(IndexFlatIP,内积=余弦) |
| LLM | **qwen-plus**(阿里云 DashScope OpenAI 兼容) |
| 切块策略 | **LLM 语义切块**(qwen-plus 给每份手册输出 sections) |

### 3. Agent 架构(5 节点全部完成 ✅)

```
START → router ─┬→ rag_search → generate ─┐
                └→ service_gen ────────────┴→ format_output → END
```

| 节点 | 状态 | 实现 |
|---|---|---|
| `router` | ✅ | 32 题 100% 准确率,qwen-plus 二分类 |
| `service_gen` | ✅ | 12 题客服话术 eval 通过 |
| `rag_search` | ✅ | 包装 LlamaIndex retriever,top-5 |
| `generate` | ✅ | qwen-plus 生成 + 抠 [[PIC:xxx]] → image_ids |
| `format_output` | ✅ | 拼成官方提交格式(JSON 数组 / 纯文本) |

### 4. 知识库

- **LLM 切块缓存**: `data/chunks_llm/*.json`(22 份手册 → 2330 个 sections,主题纯)
  - 每个 section 含 `title`、`text`(保留 `[[PIC:xxx]]` 占位符)
  - 已 commit,新机器无需重新调用 LLM 切块
- **FAISS 索引**: `index/`(不提交,新机器跑一次 `python -m src.knowledge_base` 重建,GPU 几分钟)
- **bge-m3 模型**: `models/`(不提交,~4.3GB,首次自动从 HuggingFace 下载,带镜像)

### 5. Eval 结果

**RAG 检索 eval(30 题)**:
- 手册级 Recall@1 = 96.7%, Recall@3/5 = 100%
- 块级 Recall@5 = 96.7%(LLM 标注关键词验证)

**Router eval(32 题)**: 100% 准确率

**端到端 eval(34 题)**:
- 路由准确率 34/34
- 平均耗时 4.6s/题
- 答案质量良好(产品题步骤完整、客服话术规范)

### 6. 不上 reranker 的实测原因

bge-reranker-v2-m3 加了反而让块级 Recall@1 从 76.7% → 60.0%(再加 source 上下文恢复到 66.7%,仍低于无 reranker)。

原因:cross-encoder 对短 chunk 的字面词共现过敏,把"清洁空气滤网"(空调手册)误判高于"滤网清洁步骤"(净化器手册)。
真正的产品歧义解决方案:**让 router 输出具体产品名 + rag_search metadata 过滤**(后续 Round 2 优化)。

## 待办

1. **跑全量 400 题** → 提交 A 榜 → 看真实分数
2. **Round 2 优化**(看 A 榜分数决定):
   - router 升级输出具体产品名(从 product/service → product:{产品} / service)
   - rag_search 加 metadata filter(只在该产品手册搜)
   - 解决:同主题章节跨产品手册的混淆(空调 vs 空气净化器)
3. **决赛准备**:
   - REST API(`/chat`,支持流式)
   - LangGraph checkpointer(多轮会话)

## 关键文件路径

- API key: `.env`(从 `.env.example` 复制)
- 模型缓存: `models/`(自动下载,4.3GB)
- 索引: `index/`(本地构建)
- LLM 切块缓存: `data/chunks_llm/`(已 commit)
- Eval 报告: `evals/output/`(本地生成)
