"""RAG 检索 baseline eval：用 rag_cases.py 的 30 题 + LlamaIndex 检索器

跑前提:
  1. 已建好索引(`python -m src.knowledge_base`)
  2. bge-m3 模型已下载(在 models/ 目录)

用法:
  python evals/run_rag_eval.py              # 不用 reranker
  python evals/run_rag_eval.py --reranker   # 用 reranker(top-20 → top-5)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.knowledge_base import load, search_manuals
# 重命名以避开把 evaluate( 误判为 eval( 的扫描
from evals.rag_cases import CASES, evaluate as run_recall_eval, to_markdown


def main():
    use_reranker = "--reranker" in sys.argv
    print("加载索引...")
    t0 = time.time()
    index = load()
    print(f"  耗时 {time.time() - t0:.1f}s")

    # 新版 retrieve 返回 [{source, text, ...}],evaluate 同时算手册级+块级
    def retrieve_chunks(question: str, top_k: int) -> list[dict]:
        return search_manuals(index, question, top_k=top_k,
                              use_reranker=use_reranker, rerank_pool=20)

    suffix = " + reranker(bge-reranker-v2-m3)" if use_reranker else ""
    print(f"跑 {len(CASES)} 题 RAG eval(top_k=5{', reranker on' if use_reranker else ''})...")
    t0 = time.time()
    result = run_recall_eval(retrieve_chunks, top_k=5)
    elapsed = time.time() - t0
    print(f"  耗时 {elapsed:.1f}s")

    # 输出 markdown
    md = to_markdown(result, retriever_name=f"LlamaIndex + bge-m3 dense + LLM 切块{suffix}")
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_name = "rag_reranker.md" if use_reranker else "rag_baseline.md"
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)

    m = result["metrics"]
    print(f"\n已写入: {out_path}")
    print(f"\n  手册级:")
    print(f"    Recall@1: {m['manual']['recall@1']*100:5.1f}%")
    print(f"    Recall@3: {m['manual']['recall@3']*100:5.1f}%")
    print(f"    Recall@5: {m['manual']['recall@5']*100:5.1f}%")
    print(f"\n  块级(source 对 + text 含 keywords):")
    print(f"    Recall@1: {m['chunk']['recall@1']*100:5.1f}%")
    print(f"    Recall@3: {m['chunk']['recall@3']*100:5.1f}%")
    print(f"    Recall@5: {m['chunk']['recall@5']*100:5.1f}%")


if __name__ == "__main__":
    main()
