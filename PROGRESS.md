# 项目进度

更新时间：2026-04-26

## 已完成

### 1. 数据/题目分析
- 22 份中文产品手册 + 2608 张插图 + 1 份英文汇总手册
- 400 题分布：通用客服 ~37（9%）/ 中文产品 ~162（41%）/ 英文产品 ~187（47%）/ 边界 ~14（3%）
- 提交格式：产品题 `"文本(<PIC>)", ["id1","id2"]` / 客服题 `"纯文本"`

### 2. 比赛规则
- 评分：LLM 裁判 1-5 分（看图文配合 + 回答质量 + 结构）
- 初赛：系统设计 30% + 技术实现 70%
- A 榜（现在 ~ 6/20）csv 提交 / B 榜 6/20 开放 / 决赛 8 月线下
- 当前榜首 0.7300（满分 1.0）

### 3. 技术选型
- 框架：**LangGraph**（节点+边图编排，可控、可调试、模型无关）
- 模型：**Qwen-Plus**（阿里云 DashScope，国内调用稳，单价 ¥0.0008/0.002 千 token）
- 决赛阶段再考虑加 VLM（多模态看图）

### 4. Agent 架构设计

5 节点流程：

```
START → router → ┬─ rag_search → generate → format_output → END
                 └─ service_gen ─────────→ format_output → END
```

| 节点 | 职责 | 状态 |
|------|------|------|
| router | 意图分类（产品题 / 客服题） | ⬜ 待写 |
| rag_search | FAISS 检索手册 | ⬜ 待写（复用 baseline 的 KnowledgeBase） |
| generate | 用 chunks 生成产品题答案（含 `<PIC>`） | ⬜ 待写 |
| service_gen | 直接生成客服话术（无图） | ✅ 完成 |
| format_output | 拼成官方提交格式 | ⬜ 待写 |

### 5. 代码骨架

- `src/state.py` — LangGraph State（6 字段）
- `src/hello_graph.py` — 1 节点 hello world（验证 LangGraph 工作）
- `src/nodes/service_gen.py` — service_gen 节点 + system prompt
- `evals/run_service_gen.py` — 并发跑 12 题（3 官方 + 9 csv）+ 输出 markdown
- `history_code/` — baseline 老代码（参考，已不直接使用）

### 6. service_gen 节点效果

12 题 eval（耗时 6 秒，¥0.009）：
- ✅ 风格学到位：开头"您好"、共情、具体数字、给方案、不甩锅
- ✅ 长度控制：80-130 字（接近官方）
- ✅ 纯文本：无 emoji / markdown / 列表符号
- ✅ 口语化：去掉了专业术语
- 预估 4 分稳，部分 5 分

prompt 关键设计：
- 3 个官方范例 few-shot
- 11 条风格约束（共情、数字、长度、格式、口语化等）

## 待办

按优先级：

1. **router 节点**（产品 / 客服 二分类，调一次轻量 LLM）
2. **rag_search 节点**（包装 baseline 的 `KnowledgeBase.search`）
3. **generate 节点**（产品题：用 chunks 生成回答 + 提取图片 ID）
4. **format_output 节点**（拼成 `"文本", [id]` 双段格式）
5. **agent_graph.py**（用上面 5 节点 + 边组装）
6. **跑全量 400 题** → 提交 A 榜 → 看真实分数
7. **优化方向**（按真实分数决定）：
   - 多语言 embedding（bge-m3）解决英文题
   - reranker 提升 RAG 质量
   - VLM 节点（决赛用）

## 当前已知问题 / 决策

- **API key 通过 .env 读取**，不提交到 git。新机器需复制 .env.example 为 .env 并填 key。
- **赛题数据不提交**（134MB 太大）。新机器需从官网重新下载。
- **FAISS 索引不提交**（生成后约 1.5MB，但避免与代码耦合，新机器跑一次构建脚本即可）。
- **Embedding 模型不提交**（在 `models/BAAI/`，新机器自动下载）。

## 关键文件路径

- API key：`D:\code\server\PROFILE.md`（开发机本地）
- 赛题数据：`data/`（不提交）
- 现有 FAISS 索引：`index/faiss.index`（不提交）
- baseline 旧代码：`history_code/`（已提交参考）
