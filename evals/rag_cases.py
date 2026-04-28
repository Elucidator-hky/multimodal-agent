"""RAG 检索测试用例 —— 框架无关

每个用例:
  question:      题目原文(来自 question-public.csv)
  expected:      应命中的手册文件名列表(必须命中其一)
  keywords:      chunk 文本必须同时包含的关键词(块级命中标准)
  notes:         备注

两种命中粒度(都跑):
  - manual_level: top-K 里有 chunk 来自 expected 手册
  - chunk_level:  top-K 里有 chunk 满足 (source ∈ expected) AND (text 含全部 keywords)
                  这才是"找到了正确的那段答案"

retrieve(question, top_k) 必须返回 [{"source": ..., "text": ..., ...}, ...]
"""
import json
import os
from typing import Callable


# 30 题:中文产品 24 + 英文产品 6
# keywords 设计原则:
#   - 不放产品名(LLM 切的章节里通常省略,会 false negative)
#   - 放"答案章节里独特的内容词"(动作、对象、型号等)
#   - 2-3 个词 AND 匹配,case-insensitive
CASES = [
    # ─── 中文产品 ───
    {"id": "64",  "question": "使用吹风机时，人员需要佩戴哪些防护装备？",
     "expected": ["吹风机手册.txt"],
     "keywords": ["佩戴", "防护"],
     "notes": "防护装备章节"},
    {"id": "74",  "question": "如何用空调快速调节室内温度？",
     "expected": ["空调手册.txt"],
     "keywords": ["快速", "温度"],
     "notes": "快速调温功能"},
    {"id": "83",  "question": "如何清洁空调的空气滤网?",
     "expected": ["空调手册.txt"],
     "keywords": ["清洁", "滤网"],
     "notes": "空调滤网清洁(关键词同净化器,但需 source = 空调手册)"},
    {"id": "86",  "question": "如何快速组装蒸汽清洁机？",
     "expected": ["蒸汽清洁机手册.txt"],
     "keywords": ["组装"],
     "notes": "组装章节"},
    {"id": "89",  "question": "组装人体工学椅涉及哪些部件？",
     "expected": ["人体工学椅手册.txt"],
     "keywords": ["组装", "部件"],
     "notes": "组装部件清单"},
    {"id": "95",  "question": "如何为洗碗机添加洗涤剂？",
     "expected": ["洗碗机手册.txt"],
     "keywords": ["洗涤剂"],
     "notes": "洗涤剂添加章节(独特词)"},
    {"id": "109", "question": "如何清洁空气净化器的滤网？",
     "expected": ["空气净化器手册.txt"],
     "keywords": ["清洁", "滤网"],
     "notes": "净化器滤网清洁(关键词同空调,但需 source = 净化器手册)"},
    {"id": "113", "question": "如何了解这款健身单车的技术规格？",
     "expected": ["健身单车手册.txt"],
     "keywords": ["规格"],
     "notes": "技术规格章节"},
    {"id": "124", "question": "我的DCB101型号电钻指示灯闪烁时，这些闪烁标识代表什么含义？",
     "expected": ["电钻手册.txt"],
     "keywords": ["DCB101", "闪烁"],
     "notes": "DCB101 指示灯章节(型号是关键独特词)"},
    {"id": "130", "question": "电钻的三年有限保修包含哪些内容？",
     "expected": ["电钻手册.txt"],
     "keywords": ["保修", "三年"],
     "notes": "三年保修条款"},
    {"id": "131", "question": "购买健身追踪器后，包装盒里应该有什么？",
     "expected": ["健身追踪器手册.txt"],
     "keywords": ["包装"],
     "notes": "包装盒内容"},
    {"id": "145", "question": "使用冰箱冰柜时需要注意什么？只需告诉我手册中的前五条。",
     "expected": ["冰箱手册.txt"],
     "keywords": ["冰柜"],
     "notes": "冰柜使用注意(冰柜比冰箱独特)"},
    {"id": "153", "question": "考虑到燃油高度易燃且有毒，使用发电机时我需要注意什么？",
     "expected": ["发电机手册.txt"],
     "keywords": ["燃油"],
     "notes": "燃油安全章节"},
    {"id": "168", "question": "为了正确更换发电机的发动机机油，我需要遵循的前六个步骤是什么？",
     "expected": ["发电机手册.txt"],
     "keywords": ["机油", "更换"],
     "notes": "机油更换步骤"},
    {"id": "174", "question": "如何在深水中登上并平衡摩托艇，确保操作的安全性和稳定性？",
     "expected": ["摩托艇手册.txt"],
     "keywords": ["深水"],
     "notes": "深水登艇(独特场景词)"},
    {"id": "182", "question": "水泵的核心部件有哪些？",
     "expected": ["水泵手册.txt"],
     "keywords": ["核心", "部件"],
     "notes": "核心部件章节"},
    {"id": "192", "question": "如何安全更换温控器的电池，确保更换后设备功能正常？",
     "expected": ["可编程温控器手册.txt"],
     "keywords": ["电池", "更换"],
     "notes": "电池更换章节"},
    {"id": "199", "question": "使用和操作VR头显时应采取哪些安全预防措施，以确保用户安全和设备使用寿命？",
     "expected": ["VR头显手册.txt"],
     "keywords": ["预防"],
     "notes": "安全预防章节"},
    {"id": "204", "question": "如何将功能键盘搭配CAM软件使用？",
     "expected": ["功能键盘手册.txt"],
     "keywords": ["CAM"],
     "notes": "CAM 软件配合(CAM 是独特独特词)"},
    {"id": "207", "question": "如何安装儿童电动摩托车的前轮?",
     "expected": ["儿童电动摩托车手册.txt"],
     "keywords": ["前轮"],
     "notes": "前轮安装(独特词)"},
    {"id": "213", "question": "如何查看蓝牙激光鼠标的电量状态?",
     "expected": ["蓝牙激光鼠标手册.txt"],
     "keywords": ["电量"],
     "notes": "电量查看"},
    {"id": "217", "question": "如何安装烤箱门?",
     "expected": ["烤箱手册.txt"],
     "keywords": ["安装", "门"],
     "notes": "烤箱门安装"},
    {"id": "222", "question": "如何使用烤箱的烤架?",
     "expected": ["烤箱手册.txt"],
     "keywords": ["烤架"],
     "notes": "烤架使用"},
    {"id": "232", "question": "如何设置混合即时相机的自动打印模式?",
     "expected": ["相机手册.txt"],
     "keywords": ["自动", "打印"],
     "notes": "自动打印模式"},

    # ─── 英文产品(关键词小写,case-insensitive 匹配)───
    {"id": "241", "question": "If this is the first time to use airfryer, What should I do before first use?",
     "expected": ["汇总英文手册.txt"],
     "keywords": ["before first use"],
     "notes": "首次使用前 — before first use 是答案章节标题用语"},
    {"id": "244", "question": "How the ship steers?",
     "expected": ["汇总英文手册.txt", "摩托艇手册.txt"],
     "keywords": ["steer"],
     "notes": "船的转向章节(中英文都行,steer/转向)"},
    {"id": "250", "question": "How do I use the jet wash function to clean the boat after using it?",
     "expected": ["汇总英文手册.txt", "摩托艇手册.txt"],
     "keywords": ["jet wash"],
     "notes": "jet wash 是独特词"},
    {"id": "255", "question": "When I am sailing, how do I check the engine oil level to ensure continued sailing?",
     "expected": ["汇总英文手册.txt", "摩托艇手册.txt"],
     "keywords": ["engine oil"],
     "notes": "engine oil level"},
    {"id": "265", "question": "How to use the energy saving mode of a coffee machine?",
     "expected": ["汇总英文手册.txt"],
     "keywords": ["energy saving"],
     "notes": "节能模式"},
    {"id": "268", "question": "How should I do if I want to empty the system before not in use, for frost protection or before maintenance?",
     "expected": ["汇总英文手册.txt"],
     "keywords": ["empty"],
     "notes": "排空系统(empty)"},
]


# ─────── 自动覆盖:如果存在 rag_keywords.json,用 LLM 标注的关键词 ───────
_KEYWORDS_PATH = os.path.join(os.path.dirname(__file__), "rag_keywords.json")
if os.path.exists(_KEYWORDS_PATH):
    with open(_KEYWORDS_PATH, "r", encoding="utf-8") as _f:
        _llm_data = json.load(_f)
    _kw_map = _llm_data.get("keywords_by_id", {})
    for _case in CASES:
        if _case["id"] in _kw_map and _kw_map[_case["id"]]:
            _case["keywords"] = _kw_map[_case["id"]]


# ─────── 评估函数 ───────

def _chunk_match(chunk: dict, expected_sources: list[str], keywords: list[str]) -> bool:
    """块级命中:source 在 expected AND text 含全部 keywords(case-insensitive)"""
    if chunk.get("source") not in expected_sources:
        return False
    text_lower = chunk.get("text", "").lower()
    return all(kw.lower() in text_lower for kw in keywords)


def _manual_match(chunk: dict, expected_sources: list[str]) -> bool:
    """手册级命中:source 在 expected 即可"""
    return chunk.get("source") in expected_sources


def evaluate(retrieve: Callable[[str, int], list[dict]], top_k: int = 5) -> dict:
    """同时算 manual_level + chunk_level Recall@1/3/5

    retrieve(question, top_k) 必须返回 [{"source": ..., "text": ..., ...}, ...]
    """
    assert top_k >= 5, "top_k 至少 5 才能算 Recall@5"
    rows = []
    for case in CASES:
        retrieved = retrieve(case["question"], top_k)

        manual_hits = {}
        chunk_hits = {}
        for k in (1, 3, 5):
            window = retrieved[:k]
            manual_hits[k] = any(_manual_match(c, case["expected"]) for c in window)
            chunk_hits[k] = any(
                _chunk_match(c, case["expected"], case.get("keywords", []))
                for c in window
            )

        rows.append({
            "case": case,
            "retrieved": retrieved,
            "manual_hits": manual_hits,
            "chunk_hits": chunk_hits,
        })

    total = len(rows)
    metrics = {
        "manual": {f"recall@{k}": sum(1 for r in rows if r["manual_hits"][k]) / total
                   for k in (1, 3, 5)},
        "chunk": {f"recall@{k}": sum(1 for r in rows if r["chunk_hits"][k]) / total
                  for k in (1, 3, 5)},
    }
    return {"rows": rows, "metrics": metrics, "total": total}


def to_markdown(result: dict, retriever_name: str = "") -> str:
    rows = result["rows"]
    m = result["metrics"]
    total = result["total"]

    lines = [
        f"# RAG 检索 eval{' — ' + retriever_name if retriever_name else ''}",
        "",
        f"**用例总数**: {total}",
        "",
        "| 粒度 | Recall@1 | Recall@3 | Recall@5 |",
        "|---|---|---|---|",
        f"| 手册级 | {m['manual']['recall@1']*100:.1f}% | {m['manual']['recall@3']*100:.1f}% | {m['manual']['recall@5']*100:.1f}% |",
        f"| **块级** | **{m['chunk']['recall@1']*100:.1f}%** | **{m['chunk']['recall@3']*100:.1f}%** | **{m['chunk']['recall@5']*100:.1f}%** |",
        "",
        "块级 = top-K 里有 chunk 满足 `source 在 expected` 且 `text 同时包含全部 keywords`",
        "",
        "---",
        "",
        "## 块级失败用例(Recall@5 未命中)",
        "",
    ]
    failed = [r for r in rows if not r["chunk_hits"][5]]
    if not failed:
        lines.append("(无失败)")
    else:
        for r in failed:
            c = r["case"]
            lines += [
                f"### [id={c['id']}] keywords={c['keywords']}, expected={c['expected']}",
                f"> {c['question']}",
                "",
                "Top-5:",
            ]
            for i, ck in enumerate(r["retrieved"][:5], 1):
                kw_hits = [kw for kw in c["keywords"] if kw.lower() in ck.get("text", "").lower()]
                src_ok = "✓" if ck.get("source") in c["expected"] else "✗"
                lines.append(f"  {i}. [{src_ok} {ck.get('source', '?')}] kw 命中 {len(kw_hits)}/{len(c['keywords'])}: {kw_hits}")
            lines.append("")

    lines += [
        "---",
        "",
        "## 全部结果",
        "",
        "| id | 期望手册 | keywords | 块@1 | 块@3 | 块@5 | 手册@1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        c = r["case"]
        ok = lambda x: "✅" if x else "❌"
        expected = " / ".join(c["expected"])[:30]
        kw = " ".join(c["keywords"])
        lines.append(
            f"| {c['id']} | {expected} | {kw} | "
            f"{ok(r['chunk_hits'][1])} | {ok(r['chunk_hits'][3])} | {ok(r['chunk_hits'][5])} | "
            f"{ok(r['manual_hits'][1])} |"
        )

    return "\n".join(lines)


# ─────── 自检:列覆盖范围 ───────
if __name__ == "__main__":
    from collections import Counter

    print(f"加载了 {len(CASES)} 个 RAG 测试用例")
    print()
    cnt = Counter(m for c in CASES for m in c["expected"])
    print(f"覆盖手册({len(cnt)} 份):")
    for manual, n in sorted(cnt.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {manual:30s} {n} 题")
    print()
    print(f"keywords 数量分布:")
    kw_counts = Counter(len(c["keywords"]) for c in CASES)
    for nk, n in sorted(kw_counts.items()):
        print(f"  {nk} 个关键词: {n} 题")
