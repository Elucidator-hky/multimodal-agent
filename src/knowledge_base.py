"""LlamaIndex 版知识库：解析 22 份手册 → 分块 → 向量索引

核心思路:
  1. 把 <PIC> 替换成内嵌占位符 [[PIC:filename.png]]，让 LlamaIndex 全程当普通字符
  2. 用 LlamaIndex 的 Document / SentenceSplitter / FaissVectorStore 标准流程建索引
  3. 检索时 chunk.text 自带占位符，generate 节点 regex 抠出图片名

每个 Document 的 metadata = {"source": "电钻手册.txt"} —— 评估和 metadata 过滤都靠它。
"""
import json
import os
import re
import sys
from typing import Iterable

import faiss

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─────────── 手册解析 ───────────

def _fix_invalid_escape(text: str) -> str:
    """字符级扫描修复非法 JSON 转义。

    合法转义对(\\n \\t \\r \\b \\f \\" \\\\ \\/ \\uXXXX)整体保留;
    孤立或非法转义(如 \\* \\# \\c)的反斜杠加倍 → 让 JSON 解析时还原为字面 \\。

    比正则版稳:不会把 \\\\circ 这种合法 \\\\ + 普通字符破坏成 \\\\\\circ。
    """
    out = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == '\\' and i + 1 < n and text[i + 1] in 'ntrbf"/\\u':
            out.append(text[i:i + 2])
            i += 2
        elif c == '\\':
            out.append('\\\\')
            i += 1
        else:
            out.append(c)
            i += 1
    return ''.join(out)


def parse_manual(filepath: str) -> list[tuple[str, list[str]]]:
    """解析一份手册文件，返回 [(text, image_refs), ...]

    手册是 JSON 数组格式: [text, [img1, img2, ...]]
    汇总英文手册包含多个 JSON 对象，要逐个 raw_decode。
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    content = _fix_invalid_escape(content)

    results = []
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(content):
        while pos < len(content) and content[pos] in " \t\n\r":
            pos += 1
        if pos >= len(content):
            break
        try:
            obj, end = decoder.raw_decode(content, pos)
            if isinstance(obj, list) and len(obj) == 2 and isinstance(obj[0], str):
                results.append((obj[0], obj[1] if isinstance(obj[1], list) else []))
            pos = end
        except json.JSONDecodeError:
            break
    return results


def replace_pic_with_placeholder(text: str, image_refs: list[str]) -> str:
    """把 text 中第 i 个 <PIC> 替换为 [[PIC:image_refs[i]]]。

    位置不再依赖外部数组，每个 chunk 自己带完整图片信息。
    分块时 SentenceSplitter 把 [[PIC:xxx]] 当普通文本处理，不会被破坏。
    """
    pic_iter = iter(image_refs)
    def replacer(_match):
        try:
            return f"[[PIC:{next(pic_iter)}]]"
        except StopIteration:
            return "<PIC>"  # image_refs 不够时保留原样
    return re.sub(r"<PIC>", replacer, text)


def extract_pic_filenames(text: str) -> list[str]:
    """从 chunk.text 里抠出所有 [[PIC:xxx]] 占位符的文件名。

    generate 节点回填图片 ID 时调这个。
    """
    return re.findall(r"\[\[PIC:([^\]]+)\]\]", text)


# ─────────── Document 加载 ───────────

def load_manuals_as_documents() -> list[Document]:
    """读所有手册 → Document 列表。

    每个 Document:
      - text: 整份手册（同一份手册若有多个 JSON,每个独立成 Document）
      - metadata: {"source": "电钻手册.txt"}, sub_index?: int
    """
    docs: list[Document] = []
    kb_dir = config.KB_DIR

    for filename in sorted(os.listdir(kb_dir)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(kb_dir, filename)

        parsed = parse_manual(filepath)
        if not parsed:
            print(f"  [警告] {filename} 解析为空")
            continue

        for idx, (text, image_refs) in enumerate(parsed):
            text_with_pics = replace_pic_with_placeholder(text, image_refs)
            metadata = {"source": filename}
            if len(parsed) > 1:
                metadata["sub_index"] = idx
            docs.append(Document(text=text_with_pics, metadata=metadata))

        print(f"  解析 {filename}: {len(parsed)} 段")

    return docs


# ─────────── 从 LLM 切块缓存加载 Nodes ───────────

def load_nodes_from_cache() -> list[TextNode]:
    """从 data/chunks_llm/*.json 读 LLM 切好的 sections，产出 TextNode 列表。

    每个 section 成一个 Node,metadata = {source, title, sub_index?}
    跳过 NodeParser,直接用 LLM 切好的语义节作为最终 chunk。
    """
    cache_dir = os.path.join(config.DATA_DIR, "chunks_llm")
    if not os.path.exists(cache_dir):
        raise FileNotFoundError(
            f"切块缓存不存在: {cache_dir}\n"
            f"请先跑 `python scripts/cache_all_chunks.py`"
        )

    nodes = []
    for filename in sorted(os.listdir(cache_dir)):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(cache_dir, filename), "r", encoding="utf-8") as f:
            cached = json.load(f)

        manual_filename = filename.replace(".json", ".txt")
        for part in cached["parts"]:
            for section in part["sections"]:
                metadata = {
                    "source": manual_filename,
                    "title": section.get("title", ""),
                }
                if cached.get("n_parts", 1) > 1:
                    metadata["sub_index"] = part["sub_index"]
                nodes.append(TextNode(
                    text=section["text"],
                    metadata=metadata,
                ))
    return nodes


# ─────────── 索引构建 / 加载 ───────────

def _setup_settings():
    """配置 LlamaIndex 全局 Settings"""
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=config.EMBEDDING_MODEL,
        cache_folder=config.MODELS_DIR,
        # bge-m3 不需要 query/passage 前缀(模型已 instruction-tuned)
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50,
    )
    # 不用 LLM(我们不用 query_engine,自己在 generate 节点调 LLM)
    Settings.llm = None


def build_and_save() -> VectorStoreIndex:
    """构建索引并持久化到 config.INDEX_DIR

    流程:从 data/chunks_llm/*.json 加载 LLM 切好的 sections → TextNode → embedding → FAISS
    """
    print(f"[1/4] 配置 embedding ({config.EMBEDDING_MODEL})")
    _setup_settings()

    print(f"[2/4] 加载 LLM 切块缓存")
    nodes = load_nodes_from_cache()
    print(f"  共 {len(nodes)} 个 Node(来自 {len({n.metadata['source'] for n in nodes})} 份手册)")

    print(f"[3/4] 构建 FAISS 索引(embedding)")
    faiss_index = faiss.IndexFlatIP(config.EMBED_DIM)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 直接传 nodes,跳过 NodeParser,LLM 切好的就是最终 chunk
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True,
    )

    print(f"[4/4] 持久化到 {config.INDEX_DIR}")
    os.makedirs(config.INDEX_DIR, exist_ok=True)
    index.storage_context.persist(persist_dir=config.INDEX_DIR)

    print(f"完成! 共 {faiss_index.ntotal} 个向量(每个 {config.EMBED_DIM} 维)")
    return index


def load() -> VectorStoreIndex:
    """从持久化目录加载索引"""
    if not os.path.exists(config.INDEX_DIR):
        raise FileNotFoundError(
            f"索引目录不存在: {config.INDEX_DIR}\n"
            f"请先跑 `python -m src.knowledge_base` 构建索引"
        )
    _setup_settings()
    vector_store = FaissVectorStore.from_persist_dir(config.INDEX_DIR)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=config.INDEX_DIR,
    )
    return load_index_from_storage(storage_context=storage_context)


# ─────────── 检索接口(给 rag_search 节点用) ───────────

def search_manuals(
    index: VectorStoreIndex,
    query: str,
    top_k: int = 5,
    use_reranker: bool = False,
    rerank_pool: int = 20,
) -> list[dict]:
    """检索 top-k chunk,返回结构化结果。

    参数:
      use_reranker: True 时先 retrieve top-rerank_pool,再用 bge-reranker-v2-m3 重排成 top_k
      rerank_pool: 重排前的候选池大小(只在 use_reranker=True 时生效)

    返回每项: {
        "text": chunk 文本(含 [[PIC:xxx]] 占位符),
        "source": 手册文件名,
        "score": 相似度分(若开 reranker, 这里是 rerank 分),
        "raw_score": 原 dense embedding 分,
        "pics": [图片文件名, ...],
        "title": chunk 标题(如有),
    }
    """
    actual_top = rerank_pool if use_reranker else top_k
    retriever = index.as_retriever(similarity_top_k=actual_top)
    nodes = retriever.retrieve(query)
    results = []
    for n in nodes:
        text = n.node.text
        results.append({
            "text": text,
            "source": n.node.metadata.get("source", ""),
            "title": n.node.metadata.get("title", ""),
            "score": float(n.score) if n.score is not None else 0.0,
            "raw_score": float(n.score) if n.score is not None else 0.0,
            "pics": extract_pic_filenames(text),
        })

    if use_reranker:
        from src.reranker import rerank_chunks
        results = rerank_chunks(query, results, top_n=top_k)
        # 用 rerank_score 替换 score
        for r in results:
            r["score"] = r.get("rerank_score", r["raw_score"])

    return results


# ─────────── 命令行入口 ───────────

if __name__ == "__main__":
    build_and_save()
