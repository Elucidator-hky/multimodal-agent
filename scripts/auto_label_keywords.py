"""让 qwen-plus 给每题自动提关键词:
   1. 用现有 RAG retrieve top-15 chunks
   2. 过滤到只属于 expected 手册的 chunks
   3. 让 LLM 选出真正含答案的 chunk + 从该 chunk 提取 2-3 个独特词
   4. 输出 evals/rag_keywords.json,evaluate 时优先读这个

跑完成本约 ¥0.05,耗时几十秒(并发 10 路)
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from openai import OpenAI
from src.knowledge_base import load, search_manuals
from evals.rag_cases import CASES


client = OpenAI(api_key=config.QWEN_API_KEY, base_url=config.QWEN_BASE_URL)
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "evals", "rag_keywords.json")


PROMPT = """你是 RAG 评估标注助手。下面是一道用户问题和 N 个候选 chunk(都来自相关手册)。

任务:
1. 选出最能回答问题的 chunk(返回序号 1-N)
2. 从该 chunk text 中提取 **2-3 个独特词或短语** 用于检索验证
   - 必须实际出现在该 chunk text 里
   - 必须独特,避免泛词("清洁"、"安全"、"使用"这种太宽泛)
   - 选具体动作 / 对象 / 型号 / 场景词

输出 JSON: {{"best_chunk_idx": 序号, "keywords": ["...", "..."]}}

问题: {question}

候选 chunks:
{chunks_text}
"""


def label_one_case(case: dict, retrieve_fn) -> dict:
    """对一个 case 自动提取 keywords"""
    chunks = retrieve_fn(case["question"], top_k=15)
    relevant = [c for c in chunks if c["source"] in case["expected"]][:5]

    if not relevant:
        return {"id": case["id"], "error": "no chunk from expected manual",
                "keywords": case.get("keywords", [])}

    chunks_text = "\n\n".join(
        f"[chunk {i+1}] (from {c['source']})\n{c['text'][:1000]}"
        for i, c in enumerate(relevant)
    )

    try:
        resp = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": PROMPT.format(
                question=case["question"], chunks_text=chunks_text)}],
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=512,
        )
        data = json.loads(resp.choices[0].message.content)
        kws = data.get("keywords", [])
        # 过滤:确保 kw 实际在选中 chunk 中
        idx = max(1, min(data.get("best_chunk_idx", 1), len(relevant))) - 1
        chosen = relevant[idx]["text"].lower()
        kws_verified = [kw for kw in kws if kw.lower() in chosen]

        return {
            "id": case["id"],
            "question": case["question"],
            "old_keywords": case.get("keywords", []),
            "new_keywords": kws_verified if kws_verified else kws,
            "chosen_chunk_idx": idx + 1,
            "chosen_source": relevant[idx]["source"],
            "chosen_text_preview": relevant[idx]["text"][:200],
        }
    except Exception as e:
        return {"id": case["id"], "error": str(e),
                "keywords": case.get("keywords", [])}


def main():
    print("加载索引...")
    index = load()

    def retrieve_fn(q, top_k):
        return search_manuals(index, q, top_k=top_k)

    print(f"对 {len(CASES)} 题并发标注关键词(qwen-plus, 10 路)...")
    t0 = time.time()
    results = [None] * len(CASES)
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(label_one_case, c, retrieve_fn): i for i, c in enumerate(CASES)}
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()

    print(f"耗时 {time.time() - t0:.0f}s\n")

    # 输出对比
    print(f"{'id':>4} {'old keywords':>20} → {'new keywords':<30} chunk")
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['id']:>4} ❌ {r['error']}")
            continue
        old = " ".join(r["old_keywords"])[:18]
        new = " ".join(r["new_keywords"])[:28]
        print(f"{r['id']:>4} {old:>20} → {new:<30} (chunk {r['chosen_chunk_idx']})")

    # 写 JSON(给 rag_cases.py 加载)
    keywords_map = {r["id"]: r.get("new_keywords") or r.get("keywords", []) for r in results}
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"keywords_by_id": keywords_map, "details": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\n已写入: {OUT_PATH}")


if __name__ == "__main__":
    main()
