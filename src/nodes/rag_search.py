"""rag_search 节点：bge-m3 + LlamaIndex 检索 top-5 chunks 写进 state["chunks"]

输入:  state["question"]
输出:  state["chunks"] = [{text, source, title, score, pics}, ...]

为避免重复加载索引(每次加载耗时 ~12s + 2GB GPU 显存),用模块级单例缓存。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.knowledge_base import load, search_manuals
from src.state import State


_index = None


def _get_index():
    """懒加载 + 全局单例"""
    global _index
    if _index is None:
        _index = load()
    return _index


def rag_search(state: State) -> dict:
    """节点:检索 top-5 chunks"""
    chunks = search_manuals(_get_index(), state["question"], top_k=5)
    return {"chunks": chunks}


# ─────── 单测 ───────
if __name__ == "__main__":
    test_questions = [
        "如何清洁空气净化器的滤网？",
        "我的DCB101型号电钻指示灯闪烁时代表什么？",
        "How to use the energy saving mode of a coffee machine?",
    ]
    for q in test_questions:
        state: State = {
            "question": q, "question_type": "", "chunks": [],
            "answer_text": "", "image_ids": [], "final_output": "",
        }
        update = rag_search(state)
        print(f"\n{'='*70}\nQ: {q}")
        for i, c in enumerate(update["chunks"], 1):
            preview = c["text"][:80].replace("\n", " ")
            print(f"  {i}. [{c['source']}] {c.get('title', '')[:30]} score={c['score']:.3f}")
            print(f"     pics={c['pics']}")
            print(f"     {preview}...")
