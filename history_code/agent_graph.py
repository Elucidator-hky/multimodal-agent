"""LangGraph 版客服 agent: router + RAG + 生成 + 格式化"""
import os
import sys
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.knowledge_base import KnowledgeBase, Chunk


# ────────────────────────────── 全局资源 ──────────────────────────────

client = OpenAI(api_key=config.QWEN_API_KEY, base_url=config.QWEN_BASE_URL)

kb = KnowledgeBase()
kb.load()


# ────────────────────────────── State 定义 ──────────────────────────────

class State(TypedDict):
    question: str           # 用户问题
    question_type: str      # router 写: "product" / "service"
    chunks: list            # rag_search 写: [(Chunk, score), ...]
    answer_text: str        # generate / service_gen 写: 回答正文
    image_ids: list         # generate 写: 图片 ID 列表
    final_output: str       # format_output 写: 官方格式输出


# ────────────────────────────── 节点 ──────────────────────────────

def router(state: State) -> dict:
    """判断问题类型: product (查手册) / service (客服话术)"""
    prompt = f"""判断下面用户问题属于哪一类，只返回单词 product 或 service：
- product: 产品功能、使用、操作、故障、参数等需要查产品说明书的问题
- service: 退换货、物流、发票、售后、投诉等通用客服问题

问题：{state['question']}"""

    resp = client.chat.completions.create(
        model=config.QWEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,
    )
    qtype = resp.choices[0].message.content.strip().lower()
    if "service" in qtype:
        qtype = "service"
    else:
        qtype = "product"
    return {"question_type": qtype}


def rag_search(state: State) -> dict:
    """检索手册"""
    results = kb.search(state["question"], top_k=config.TOP_K)
    return {"chunks": results}


def generate(state: State) -> dict:
    """产品题: 用 chunks 生成回答"""
    contexts = []
    all_image_ids: list = []
    for chunk, score in state["chunks"]:
        ctx = f"【来源: {chunk.source_file}】\n{chunk.text}"
        if chunk.image_refs:
            ctx += f"\n[图片映射: {chunk.image_refs}]"
            all_image_ids.extend(chunk.image_refs)
        contexts.append(ctx)
    context_text = "\n---\n".join(contexts)

    prompt = f"""你是产品客服。基于知识库回答用户问题。
- 回答中保留 <PIC> 标记（每个 <PIC> 对应一张图）
- 详细、分点、结构清晰
- 用户用什么语言提问就用什么语言回答

用户问题：{state['question']}

知识库内容：
{context_text}"""

    resp = client.chat.completions.create(
        model=config.QWEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2048,
    )
    answer = resp.choices[0].message.content
    pic_count = answer.count("<PIC>")
    used_ids = all_image_ids[:pic_count] if pic_count else []
    return {"answer_text": answer, "image_ids": used_ids}


def service_gen(state: State) -> dict:
    """客服题: 直接生成客服话术，无图"""
    prompt = f"""你是友好专业的电商客服。用户咨询了通用客服类问题（退换货/物流/发票/售后等），
请给出得体、有帮助、口语化的回答。不要编造具体政策细节，可以建议联系客服核实。

用户问题：{state['question']}"""

    resp = client.chat.completions.create(
        model=config.QWEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=512,
    )
    return {"answer_text": resp.choices[0].message.content, "image_ids": []}


def format_output(state: State) -> dict:
    """拼成官方格式: "文本", ["id1", "id2"]  / 没图片的话只有 "文本" """
    text = state["answer_text"]
    ids = state.get("image_ids", [])
    if ids:
        formatted = f'"{text}", {ids}'
    else:
        formatted = f'"{text}"'
    return {"final_output": formatted}


# ────────────────────────────── 构图 ──────────────────────────────

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
        lambda s: s["question_type"],
        {"product": "rag_search", "service": "service_gen"},
    )
    g.add_edge("rag_search", "generate")
    g.add_edge("generate", "format_output")
    g.add_edge("service_gen", "format_output")
    g.add_edge("format_output", END)

    return g.compile()


app = build_graph()


# ────────────────────────────── 测试入口 ──────────────────────────────

if __name__ == "__main__":
    tests = [
        "电钻指示灯闪烁是什么意思？",
        "我想退货，但是已经超过7天无理由退换货期限了，还能退吗？",
        "VR 头显的游玩区域推荐尺寸是多少？",
        "How to clean the air purifier filter?",
        "你们的物流支持送到乡镇吗？",
    ]
    for i, q in enumerate(tests, 1):
        print(f"\n{'='*60}\n[{i}] Q: {q}")
        result = app.invoke({"question": q})
        print(f"路由 → {result['question_type']}")
        print(f"输出（前 300 字）:\n{result['final_output'][:300]}")
