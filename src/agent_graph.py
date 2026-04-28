"""LangGraph 整图组装:5 节点 + router 条件边

流程:
  START → router → ┬─ rag_search → generate → format_output → END  (产品题)
                   └─ service_gen ─────────→ format_output → END     (客服题)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import END, START, StateGraph
from src.state import State
from src.nodes.router import router
from src.nodes.service_gen import service_gen
from src.nodes.rag_search import rag_search
from src.nodes.generate import generate
from src.nodes.format_output import format_output


def _route_by_type(state: State) -> str:
    """router 条件边:根据 question_type 选下一节点"""
    return state["question_type"]


def build_graph():
    g = StateGraph(State)

    g.add_node("router", router)
    g.add_node("rag_search", rag_search)
    g.add_node("generate", generate)
    g.add_node("service_gen", service_gen)
    g.add_node("format_output", format_output)

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router",
        _route_by_type,
        {"product": "rag_search", "service": "service_gen"},
    )
    g.add_edge("rag_search", "generate")
    g.add_edge("generate", "format_output")
    g.add_edge("service_gen", "format_output")
    g.add_edge("format_output", END)

    return g.compile()


app = build_graph()


# ─────── 端到端测试 ───────
if __name__ == "__main__":
    from src.tracing import run_with_trace

    test_cases = [
        # 产品题(中文)
        ("如何清洁空气净化器的滤网？", "product"),
        # 产品题(英文)
        ("How to use the energy saving mode of a coffee machine?", "product"),
        # 客服题
        ("请问你们家的商品支持7天无理由退换货吗？", "service"),
        # 边界(像产品但本质客服)
        ("我购买的电钻不到一个月就坏了，要求退货退款怎么处理？", "service"),
    ]

    for q, expected_type in test_cases:
        print(f"\n{'#' * 75}")
        print(f"# Q: {q}")
        print(f"# Expected: {expected_type}")
        print('#' * 75)

        run_with_trace(app, {
            "question": q, "question_type": "", "chunks": [],
            "answer_text": "", "image_ids": [], "final_output": "",
        })
