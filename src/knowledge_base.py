"""知识库解析、分块与向量索引构建"""
import json
import os
import re
import pickle
from dataclasses import dataclass, field
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


@dataclass
class Chunk:
    """文档块"""
    text: str                           # 块文本内容
    title: str                          # 所属标题
    source_file: str                    # 来源文件名
    image_refs: list = field(default_factory=list)  # 该块涉及的图片引用列表
    pic_positions: dict = field(default_factory=dict)  # PIC序号 -> 图片文件名映射


def parse_manual(filepath: str) -> list:
    """
    解析手册文件，返回 [(text, image_refs), ...] 列表。
    汇总英文手册包含多个 JSON 对象，需要逐个解析。
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # 修复常见的无效 JSON 转义（如 \* \#  等）
    # 保留合法转义：\n \t \r \\ \" \/ \b \f \uXXXX
    content = re.sub(r'\\(?![ntrbf\\/\"u])', r'\\\\', content)

    results = []
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(content):
        # 跳过空白
        while pos < len(content) and content[pos] in " \t\n\r":
            pos += 1
        if pos >= len(content):
            break
        try:
            obj, end = decoder.raw_decode(content, pos)
            if isinstance(obj, list) and len(obj) == 2:
                results.append((obj[0], obj[1]))
            pos = end
        except json.JSONDecodeError:
            break

    return results


def build_pic_mapping(text: str, image_refs: list) -> dict:
    """
    构建 <PIC> 序号到图片文件名的映射。
    text 中第 i 个 <PIC> 对应 image_refs[i]。
    返回 {pic_index: image_filename} 字典。
    """
    mapping = {}
    pic_count = 0
    for m in re.finditer(r"<PIC>", text):
        if pic_count < len(image_refs):
            mapping[pic_count] = image_refs[pic_count]
        pic_count += 1
    return mapping


def split_by_titles(text: str, pic_mapping: dict, source_file: str) -> list[Chunk]:
    """
    按 # 标题分块。
    如果某块 > 1500 字，按 <PIC> 标记进一步分割。
    """
    lines = text.split("\n")
    chunks = []
    current_title = source_file  # 默认标题用文件名
    current_lines = []
    current_pic_start = 0  # 当前块之前已出现的 PIC 数量

    # 先计算每行之前累计的 PIC 数量
    line_pic_counts = []
    total_pics = 0
    for line in lines:
        line_pic_counts.append(total_pics)
        total_pics += line.count("<PIC>")

    def make_chunk(title, chunk_lines, global_pic_offset):
        chunk_text = "\n".join(chunk_lines).strip()
        if not chunk_text:
            return None

        # 收集该块中的图片引用
        chunk_pics = []
        chunk_pic_map = {}
        local_idx = 0
        for cline in chunk_lines:
            for _ in re.finditer(r"<PIC>", cline):
                abs_idx = global_pic_offset + local_idx
                if abs_idx in pic_mapping:
                    ref = pic_mapping[abs_idx]
                    chunk_pics.append(ref)
                    chunk_pic_map[local_idx] = ref
                local_idx += 1

        return Chunk(
            text=chunk_text,
            title=title,
            source_file=source_file,
            image_refs=chunk_pics,
            pic_positions=chunk_pic_map,
        )

    for i, line in enumerate(lines):
        if line.startswith("# ") and current_lines:
            # 输出当前块
            chunk = make_chunk(current_title, current_lines, current_pic_start)
            if chunk:
                chunks.append(chunk)
            current_title = line.strip("# ").strip()
            current_pic_start = line_pic_counts[i]
            current_lines = [line]
        else:
            if line.startswith("# "):
                current_title = line.strip("# ").strip()
            current_lines.append(line)

    # 最后一块
    if current_lines:
        chunk = make_chunk(current_title, current_lines, current_pic_start)
        if chunk:
            chunks.append(chunk)

    # 对大块进行二次分割
    final_chunks = []
    for chunk in chunks:
        if len(chunk.text) > 1500:
            sub_chunks = split_large_chunk(chunk)
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)

    return final_chunks


def split_large_chunk(chunk: Chunk) -> list[Chunk]:
    """
    对大块按 <PIC> 标记进行二次分割。
    如果没有 PIC 标记，则按段落分割。
    """
    text = chunk.text
    if "<PIC>" not in text:
        # 按段落分割（双换行）
        paragraphs = re.split(r"\n\n+", text)
        sub_chunks = []
        buffer = []
        for para in paragraphs:
            buffer.append(para)
            if len("\n\n".join(buffer)) > 1200:
                sub_chunks.append(Chunk(
                    text="\n\n".join(buffer),
                    title=chunk.title,
                    source_file=chunk.source_file,
                    image_refs=[],
                    pic_positions={},
                ))
                buffer = []
        if buffer:
            sub_chunks.append(Chunk(
                text="\n\n".join(buffer),
                title=chunk.title,
                source_file=chunk.source_file,
                image_refs=[],
                pic_positions={},
            ))
        return sub_chunks if sub_chunks else [chunk]

    # 按 PIC 分割
    parts = re.split(r"(<PIC>)", text)
    sub_chunks = []
    buffer = ""
    buffer_pics = []
    pic_idx = 0

    for part in parts:
        if part == "<PIC>":
            buffer += part
            # 获取对应的图片引用
            if pic_idx < len(chunk.image_refs):
                buffer_pics.append(chunk.image_refs[pic_idx])
            pic_idx += 1

            if len(buffer) > 1200:
                sub_chunks.append(Chunk(
                    text=buffer.strip(),
                    title=chunk.title,
                    source_file=chunk.source_file,
                    image_refs=buffer_pics.copy(),
                    pic_positions={},
                ))
                buffer = ""
                buffer_pics = []
        else:
            buffer += part

    if buffer.strip():
        sub_chunks.append(Chunk(
            text=buffer.strip(),
            title=chunk.title,
            source_file=chunk.source_file,
            image_refs=buffer_pics.copy(),
            pic_positions={},
        ))

    return sub_chunks if sub_chunks else [chunk]


class KnowledgeBase:
    """知识库管理类"""

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.index: Optional[faiss.IndexFlatIP] = None
        self.model: Optional[SentenceTransformer] = None

    def load_manuals(self):
        """加载并解析所有手册"""
        print("正在加载手册...")
        kb_dir = config.KB_DIR
        all_chunks = []

        for filename in sorted(os.listdir(kb_dir)):
            if not filename.endswith(".txt"):
                continue

            filepath = os.path.join(kb_dir, filename)
            manual_name = filename.replace(".txt", "")
            print(f"  解析: {filename}")

            manuals = parse_manual(filepath)
            for idx, (text, image_refs) in enumerate(manuals):
                # 对汇总英文手册，用子序号区分
                if len(manuals) > 1:
                    src = f"{manual_name}_{idx}"
                else:
                    src = manual_name

                pic_mapping = build_pic_mapping(text, image_refs)
                chunks = split_by_titles(text, pic_mapping, src)
                all_chunks.extend(chunks)

        self.chunks = all_chunks
        print(f"总计 {len(self.chunks)} 个文档块")
        # 打印统计
        total_chars = sum(len(c.text) for c in self.chunks)
        total_pics = sum(len(c.image_refs) for c in self.chunks)
        print(f"总字符数: {total_chars}, 总图片引用: {total_pics}")

    def _get_model(self) -> SentenceTransformer:
        if self.model is None:
            print(f"正在加载 embedding 模型: {config.EMBEDDING_MODEL}")
            self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        return self.model

    def build_index(self):
        """构建 FAISS 向量索引"""
        if not self.chunks:
            self.load_manuals()

        model = self._get_model()

        # 准备文本（标题 + 内容前512字）
        texts = []
        for chunk in self.chunks:
            # 用标题+来源+内容拼接，提高检索质量
            prefix = f"{chunk.source_file} {chunk.title}"
            t = f"{prefix}\n{chunk.text[:512]}"
            texts.append(t)

        print(f"正在生成 {len(texts)} 个向量...")
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        # 构建 FAISS 内积索引（因为已归一化，内积等价于余弦相似度）
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        print(f"索引构建完成，共 {self.index.ntotal} 个向量，维度 {dim}")

    def save(self):
        """保存索引和文档块"""
        os.makedirs(config.INDEX_DIR, exist_ok=True)

        faiss.write_index(self.index, config.FAISS_INDEX_PATH)

        with open(config.CHUNKS_PATH, "wb") as f:
            pickle.dump(self.chunks, f)

        print(f"索引已保存到 {config.INDEX_DIR}")

    def load(self):
        """加载索引和文档块"""
        if not os.path.exists(config.FAISS_INDEX_PATH):
            raise FileNotFoundError(f"索引文件不存在: {config.FAISS_INDEX_PATH}")

        self.index = faiss.read_index(config.FAISS_INDEX_PATH)

        with open(config.CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)

        print(f"已加载索引: {self.index.ntotal} 个向量, {len(self.chunks)} 个文档块")

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """搜索最相关的文档块"""
        model = self._get_model()

        query_embedding = model.encode(
            [query],
            normalize_embeddings=True,
        )
        query_embedding = np.array(query_embedding, dtype=np.float32)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))

        return results


def build_and_save():
    """构建并保存索引（命令行入口）"""
    kb = KnowledgeBase()
    kb.load_manuals()
    kb.build_index()
    kb.save()
    print("完成！")


if __name__ == "__main__":
    build_and_save()
