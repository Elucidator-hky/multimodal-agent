"""端到端 eval:跑整图 → 输出 markdown 让人工评估答案质量

测试集:
  - 30 题 RAG cases(全产品题,中英都有)
  - 4 题客服题(覆盖退换货/投诉/智能客服)
共 34 题
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent_graph import app
from src.nodes.rag_search import _get_index
from evals.rag_cases import CASES


# 客服题(从 question-public.csv 挑 4 个代表)
SERVICE_CASES = [
    {"id": "1",  "question": "请问你们家的商品支持7天无理由退换货吗？需要自己承担运费吗？",
     "expected_type": "service"},
    {"id": "7",  "question": "我收到的商品和图片不一样，颜色偏差很大，我要投诉！",
     "expected_type": "service"},
    {"id": "26", "question": "请问你们的智能客服能解答哪些问题？智能客服解答不了的问题，怎么办？",
     "expected_type": "service"},
    {"id": "46", "question": "我上个月购买的家电，使用不到一个月就出现故障，联系售后寄回维修，现在已经维修15天了，还没收到，而且我发现商品是翻新机，请问该怎么处理？",
     "expected_type": "service"},
]

# 产品题(复用 RAG cases)
PRODUCT_CASES = [
    {"id": c["id"], "question": c["question"], "expected_type": "product",
     "expected_manuals": c["expected"]}
    for c in CASES
]

ALL_CASES = PRODUCT_CASES + SERVICE_CASES


def run_one(case: dict) -> dict:
    """跑一题,返回 case + 最终 state"""
    t0 = time.time()
    initial = {
        "question": case["question"], "question_type": "", "chunks": [],
        "answer_text": "", "image_ids": [], "final_output": "",
    }
    try:
        result = app.invoke(initial)
        return {**case, **result, "elapsed_s": time.time() - t0}
    except Exception as e:
        return {**case, "error": str(e), "elapsed_s": time.time() - t0}


def to_markdown(results: list) -> str:
    total = len(results)
    errors = [r for r in results if "error" in r]
    routed_correctly = sum(
        1 for r in results
        if "error" not in r and r.get("question_type") == r.get("expected_type")
    )
    avg_time = sum(r.get("elapsed_s", 0) for r in results) / total

    lines = [
        "# 端到端 eval — 5 节点 agent 完整跑",
        "",
        f"**题数**: {total} (产品 {len(PRODUCT_CASES)} + 客服 {len(SERVICE_CASES)})",
        f"**路由准确率**: {routed_correctly}/{total - len(errors)}",
        f"**平均耗时/题**: {avg_time:.1f}s",
        f"**异常**: {len(errors)} 个",
        "",
        "---",
        "",
    ]

    for r in results:
        case_id = r.get("id", "?")
        q = r.get("question", "")
        exp_type = r.get("expected_type", "?")

        if "error" in r:
            lines += [
                f"## [id={case_id}] ❌ 异常",
                f"> {q}",
                f"```\n{r['error']}\n```",
                "",
            ]
            continue

        actual_type = r.get("question_type", "?")
        route_ok = "✅" if actual_type == exp_type else "❌"
        chunks = r.get("chunks", [])
        answer = r.get("answer_text", "")
        image_ids = r.get("image_ids", [])
        final = r.get("final_output", "")

        lines += [
            f"## [id={case_id}] {exp_type} {route_ok}(实际: {actual_type}) — {r['elapsed_s']:.1f}s",
            "",
            f"**Q:** {q}",
            "",
        ]

        if chunks:
            lines.append("**检索 top-3 chunks:**")
            for i, c in enumerate(chunks[:3], 1):
                src = c.get("source", "")
                title = c.get("title", "")
                score = c.get("score", 0)
                lines.append(f"  {i}. `{src}` — {title} (score={score:.3f})")
            lines.append("")

        lines += [
            f"**生成答案** (image_ids: {image_ids}):",
            "",
            f"> {answer}",
            "",
            f"**官方格式输出:**",
            "",
            f"```\n{final}\n```",
            "",
            "---",
            "",
        ]

    return "\n".join(lines)


def main():
    print("warm up index...")
    _get_index()
    print(f"开始跑 {len(ALL_CASES)} 题(并发 8 路)\n")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run_one, ALL_CASES))
    elapsed = time.time() - t0

    md = to_markdown(results)
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "e2e_eval.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n总耗时: {elapsed:.0f}s")
    routed_correctly = sum(
        1 for r in results
        if "error" not in r and r.get("question_type") == r.get("expected_type")
    )
    errors = [r for r in results if "error" in r]
    print(f"路由准确率: {routed_correctly}/{len(results) - len(errors)}")
    print(f"异常: {len(errors)}")
    print(f"\n报告写入: {out_path}")


if __name__ == "__main__":
    main()
