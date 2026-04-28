"""bge-reranker-v2-m3 重排器:对 (query, chunk_text) 算 cross-encoder 相关度

用法:
  from src.reranker import get_reranker
  reranker = get_reranker()
  scores = reranker.compute_score([(query, text1), (query, text2), ...])

或直接用 rerank_chunks(query, chunks, top_n) 一键重排
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# 全局缓存,避免反复加载(模型 1.1GB)
_reranker = None
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def get_reranker():
    """加载 reranker(单例),用 sentence-transformers CrossEncoder 兼容性最好"""
    global _reranker
    if _reranker is None:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from sentence_transformers import CrossEncoder
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[reranker] 加载 {RERANKER_MODEL} (device={device})...")
        _reranker = CrossEncoder(
            RERANKER_MODEL,
            max_length=512,
            device=device,
            cache_dir=config.MODELS_DIR,
        )
        print("[reranker] 加载完成")
    return _reranker


def rerank_chunks(query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    """对 chunks 重排,返回前 top_n。

    chunks 是 search_manuals 输出的格式: [{"text": ..., "source": ..., "score": ..., "pics": ...}, ...]
    重排后每个 chunk 加一个 "rerank_score" 字段。
    """
    if not chunks:
        return []
    reranker = get_reranker()
    # 关键:把 source + title 加到 chunk text 前,让 reranker 知道产品上下文
    # (chunk text 里 LLM 切的章节常省略产品名,导致跨手册误判)
    pairs = []
    for c in chunks:
        src = c.get("source", "").replace(".txt", "")
        title = c.get("title", "")
        prefix = f"[{src}"
        if title:
            prefix += f" — {title}"
        prefix += "]\n"
        pairs.append([query, prefix + c["text"]])

    scores = reranker.predict(pairs, show_progress_bar=False)

    for c, s in zip(chunks, scores):
        c["rerank_score"] = float(s)

    chunks_sorted = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return chunks_sorted[:top_n]


if __name__ == "__main__":
    # 自检:跑一个 query 验证 reranker 能用
    from src.knowledge_base import load, search_manuals

    index = load()
    query = "如何清洁空气净化器的滤网?"
    print(f"\nQuery: {query}")
    print("\n=== 不用 reranker (top-5) ===")
    raw = search_manuals(index, query, top_k=5)
    for i, r in enumerate(raw, 1):
        print(f"  {i}. [{r['source']}] score={r['score']:.3f} | {r['text'][:80]}...")

    print("\n=== 用 reranker (top-20 → top-5) ===")
    raw20 = search_manuals(index, query, top_k=20)
    reranked = rerank_chunks(query, raw20, top_n=5)
    for i, r in enumerate(reranked, 1):
        print(f"  {i}. [{r['source']}] rerank={r['rerank_score']:.3f} | {r['text'][:80]}...")
